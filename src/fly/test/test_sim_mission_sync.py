"""Exercise mission callbacks without starting ROS, cameras, or an aircraft."""

import ast
from enum import Enum
import math
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'control' / 'sim' / '0707.py'


@pytest.fixture
def callbacks():
    """Load the real callback bodies with only their external services replaced."""
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'OffboardControl'
    )
    names = {
        '_start_smooth_move', '_publish_smooth_move_setpoint',
        'navigate_to_drop_area', 'control_timer_callback',
        '_copy_latest_vision_result', 'vision_timer_callback',
        '_build_recon_mission_map', 'coordinate_current_FRD2NED',
        '_capture_widecam_pose_snapshot',
    }
    methods = [
        node for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {method.name for method in methods} == names
    enums = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in ('DroppingState', 'MissionState')
    ]
    namespace = {
        'Enum': Enum,
        'math': math,
        'VehicleStatus': SimpleNamespace(NAVIGATION_STATE_OFFBOARD=14),
        'cv2': SimpleNamespace(
            putText=lambda *args: None, FONT_HERSHEY_SIMPLEX=0,
        ),
    }
    exec(compile(ast.Module(body=enums + methods, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class FakeTime:
    def __init__(self, seconds):
        self.nanoseconds = int(seconds * 1e9)

    def __sub__(self, other):
        return FakeTime((self.nanoseconds - other.nanoseconds) / 1e9)


def make_node(callbacks):
    clock = SimpleNamespace(seconds=0.0)
    clock.now = lambda: FakeTime(clock.seconds)
    logger = SimpleNamespace(**{
        level: lambda *args, **kwargs: None
        for level in ('info', 'warn', 'error')
    })
    node = SimpleNamespace(
        clock=clock,
        get_clock=lambda: clock,
        get_logger=lambda: logger,
        _frame_lock=threading.Lock(),
        _pose_lock=threading.Lock(),
        _vision_result_lock=threading.Lock(),
        _map_data_lock=threading.Lock(),
        _vision_ready_event=threading.Event(),
        is_vision_ready=True,
        latest_frame=np.zeros((4, 4, 3), dtype=np.uint8),
        latest_frame_pose_snapshot=None,
        latest_frame_sequence=1,
        last_processed_frame_sequence=0,
        latest_vision_info=[],
        latest_vision_frame_pose=None,
        latest_vision_frame_sequence=-1,
        latest_vision_mission_state=None,
        latest_annotated_frame=None,
        latest_drop_evaluation=None,
        frame_received_time=FakeTime(0.0),
        image_timeout_sec=100.0,
        global_map_accepting_samples=True,
        vehicle_local_position=SimpleNamespace(x=0.0, y=0.0, z=-2.8, heading=0.0),
        vehicle_status=SimpleNamespace(nav_state=14),
        mission_state=callbacks['MissionState'].GLOBAL_SEARCH,
        initial_z=0.0,
        log_counter=0,
        offboard_setpoint_counter=20,
        smoothing_speed=1.0,
        min_smoothing_duration=1.0,
        max_smoothing_duration=8.0,
        dt=0.5,
        enable_smooth_transit=True,
        smooth_transit_to_recon_started=False,
        is_smoothing_descent=False,
        is_ReadyToTakeoff=True,
        is_AtTakeoffHeight=True,
        is_AtDropArea=True,
        is_FinishDrop=True,
        Is_Finish_1st_Drop=True,
        Is_Finish_2nd_Drop=True,
        current_dropping_state={
            1: callbacks['DroppingState'].COMPLETED,
            2: callbacks['DroppingState'].COMPLETED,
        },
        post_drop_delay=1.0,
        timeout_2nd_postdrop=None,
        takeoff_target_height=-2.8,
        DropArea_x=1.0,
        DropArea_y=0.0,
        nav_threshold=0.2,
        forward_x=3.0,
        recon_forward_distance=7.0,
        recon_targets_ned=[],
        handle_rtl_state=lambda: False,
        coordinate_FRD2NED=lambda x, y: (x, y),
        publications=[],
    )
    node._vision_ready_event.set()
    node.publish_position_setpoint = lambda *position: node.publications.append(position)
    for name, value in callbacks.items():
        if callable(value) and name.startswith(('_', 'navigate_', 'control_', 'vision_', 'coordinate_')):
            setattr(node, name, MethodType(value, node))
    return node


def test_smooth_setpoints_start_at_current_pose_and_finish_at_endpoint(callbacks):
    node = make_node(callbacks)
    start = (0.0, 0.0, -2.8)
    end = (2.0, 0.0, -1.8)
    node._start_smooth_move(end)
    while node.is_smoothing_descent:
        node._publish_smooth_move_setpoint()

    assert node.publications[0] == start
    assert node.publications[-1] == end
    assert all(
        first[0] <= second[0] and first[2] <= second[2]
        for first, second in zip(node.publications, node.publications[1:])
    )


@pytest.mark.parametrize('enabled', [False, True])
def test_drop_area_navigation_uses_switch_and_waits_for_actual_arrival(callbacks, enabled):
    node = make_node(callbacks)
    node.is_AtDropArea = False
    node.enable_smooth_transit = enabled
    node._start_smooth_move((node.DropArea_x, node.DropArea_y, node.takeoff_target_height))
    node.navigate_to_drop_area()

    expected_x = 0.0 if enabled else node.DropArea_x
    assert node.publications[-1] == (expected_x, 0.0, -2.8)
    assert not node.is_AtDropArea

    node.vehicle_local_position.x = node.DropArea_x
    node.navigate_to_drop_area()
    assert node.is_AtDropArea
    if enabled:
        assert not node.is_smoothing_descent


def test_recon_transit_is_initialized_once_and_advances_when_arrived(callbacks):
    node = make_node(callbacks)
    node.mission_state = callbacks['MissionState'].RETURN_TO_CENTER_DROPAREA
    node.control_timer_callback()
    assert node.smooth_transit_to_recon_started
    assert node.smoothing_step_counter == 1
    start = node.smoothing_start_pos

    node.vehicle_local_position.x = 0.5
    node.control_timer_callback()
    assert node.smoothing_step_counter == 2
    assert node.smoothing_start_pos == start

    node.vehicle_local_position.x = node.forward_x + node.recon_forward_distance
    node.control_timer_callback()
    assert node.mission_state == callbacks['MissionState'].TRANSIT_TO_RECON_OFFBOARD
    assert not node.is_smoothing_descent


@pytest.mark.parametrize('state_name', ['TARGETING_CYCLE', 'TIMEOUT_DROP'])
def test_second_release_finishes_then_waits_once_before_return(callbacks, state_name):
    node = make_node(callbacks)
    node.mission_state = callbacks['MissionState'][state_name]
    node.is_FinishDrop = False
    node.Is_Finish_2nd_Drop = False
    node.current_dropping_state[2] = callbacks['DroppingState'].STEP_4_COMMANDED
    node.manage_dropping_sequence = lambda drop_number: True
    node.vehicle_local_position.z = -1.8

    node.control_timer_callback()
    assert node.Is_Finish_2nd_Drop and node.is_FinishDrop
    assert node.timeout_2nd_postdrop is not None
    assert node.mission_state == callbacks['MissionState'][state_name]

    node.clock.seconds = 0.5
    node.control_timer_callback()
    assert node.publications[-1] == (0.0, 0.0, node.takeoff_target_height)

    node.clock.seconds = 1.01
    node.control_timer_callback()
    assert node.publications[-1] == (node.DropArea_x, node.DropArea_y, node.takeoff_target_height)

    node.vehicle_local_position.x = node.DropArea_x
    node.vehicle_local_position.z = node.takeoff_target_height
    node.control_timer_callback()
    assert node.mission_state == callbacks['MissionState'].RETURN_TO_CENTER_DROPAREA


def test_vision_discards_result_if_mission_changes_during_inference(callbacks):
    node = make_node(callbacks)
    node.latest_frame_pose_snapshot = dict(x=1.0, y=2.0, z=-5.0, yaw=0.0, roll=0.0, pitch=0.0)
    samples = []
    node._collect_global_search_map_sample = lambda *args: samples.append(args)

    def infer(frame, altitude, **kwargs):
        node.mission_state = callbacks['MissionState'].RECON_SEARCH
        return [{'name': 'Left', 'coords_frd': (1.0, 0.0)}], frame

    node.vision_controller = SimpleNamespace(process_frame=infer)
    node.vision_timer_callback()
    assert node.latest_vision_info == []
    assert node.latest_vision_frame_sequence == -1
    assert samples == []


def test_vision_stores_matching_pose_and_respects_closed_map(callbacks):
    node = make_node(callbacks)
    pose = dict(x=1.0, y=2.0, z=-5.0, yaw=0.0, roll=0.0, pitch=0.0)
    node.latest_frame_pose_snapshot = pose
    targets = [{'name': 'Left', 'coords_frd': (1.0, 0.0)}]
    samples = []
    node._collect_global_search_map_sample = lambda *args: samples.append(args)

    def infer(frame, altitude, **kwargs):
        node.global_map_accepting_samples = False
        return targets, frame

    node.vision_controller = SimpleNamespace(process_frame=infer)
    node.vision_timer_callback()
    info, copied_pose, sequence = node._copy_latest_vision_result(node.mission_state)
    assert info == targets
    assert copied_pose == pose
    assert sequence == 1
    assert samples == []
    copied_pose['x'] = 100.0
    info[0]['name'] = 'changed'
    assert node.latest_vision_frame_pose == pose
    assert node.latest_vision_info == targets
    assert node._copy_latest_vision_result(callbacks['MissionState'].RECON_SEARCH) == ([], None, -1)


def test_recon_mapping_uses_frame_pose_instead_of_current_pose(callbacks):
    node = make_node(callbacks)
    node.vehicle_local_position.x = 100.0
    node.vehicle_local_position.y = 200.0
    node.vehicle_local_position.heading = 0.0
    node._build_recon_mission_map(
        [{'name': 'Recon_1', 'coords_frd': (2.0, 1.0)}],
        sample_pose=dict(x=10.0, y=20.0, yaw=math.pi / 2),
    )
    assert node.recon_targets_ned[0]['coords_ned'] == pytest.approx((9.0, 22.0))


def test_pose_snapshot_rejects_unsynchronized_attitude(callbacks):
    node = make_node(callbacks)
    node.vehicle_local_position.timestamp_sample = 1_000_000
    node.vehicle_roll = 0.1
    node.vehicle_pitch = 0.2
    node.vehicle_attitude_timestamp_us = 1_200_000
    node.widecam_pose_attitude_skew_us = 100_000
    node.widecam_local_reference_warned = False
    assert node._capture_widecam_pose_snapshot() is None

    node.vehicle_attitude_timestamp_us = 1_050_000
    snapshot = node._capture_widecam_pose_snapshot()
    assert snapshot['x'] == 0.0
    assert snapshot['roll'] == 0.1
    assert snapshot['pitch'] == 0.2


def test_model_failure_keeps_control_gated_until_success(callbacks):
    node = make_node(callbacks)
    node._vision_ready_event.clear()
    node.is_vision_ready = False
    node.offboard_heartbeat_enabled = False
    node.vision_controller = SimpleNamespace(load_model=lambda: False)
    node.vision_timer_callback()
    node.control_timer_callback()
    assert not node._vision_ready_event.is_set()
    assert not node.offboard_heartbeat_enabled
    assert node.publications == []

    node.vision_controller.load_model = lambda: True
    node.vision_timer_callback()
    assert node._vision_ready_event.is_set()
    assert node.is_vision_ready


def test_disabled_recon_transit_publishes_destination_directly(callbacks):
    node = make_node(callbacks)
    node.mission_state = callbacks['MissionState'].RETURN_TO_CENTER_DROPAREA
    node.enable_smooth_transit = False
    node.control_timer_callback()
    assert node.publications == [(10.0, 0.0, node.takeoff_target_height)]
    assert not node.smooth_transit_to_recon_started


@pytest.mark.parametrize('overall_timeout', [False, True])
def test_global_search_closes_sampling_on_normal_or_timeout_exit(callbacks, overall_timeout):
    node = make_node(callbacks)
    node.is_FinishDrop = False
    node.drop_phase_start_time = FakeTime(0.0)
    node.drop_phase_timeout = 5.0 if overall_timeout else 80.0
    node.search_start_time = FakeTime(0.0)
    node.search_timeout = 5.0
    node.global_search_height = -5.5
    node.clock.seconds = 10.0
    averaged = []

    def average_map():
        assert not node.global_map_accepting_samples
        averaged.append(True)
        node.mission_targets_ned = [{'name': 'Left', 'coords_ned': (2.0, 3.0)}]

    node._calculate_and_store_average_map = average_map
    node.control_timer_callback()
    assert not node.global_map_accepting_samples
    if overall_timeout:
        assert node.mission_state == callbacks['MissionState'].TIMEOUT_DROP
        assert averaged == []
    else:
        assert node.mission_state == callbacks['MissionState'].TARGETING_CYCLE
        assert averaged == [True]
