"""Exercise real recon callbacks with deterministic camera and tracker fakes."""

import ast
from collections import Counter, deque
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest


CONTROL_DIR = Path(__file__).resolve().parents[1] / 'control'


def _load_class_methods(path, class_name, method_names, namespace):
    """Load only the tested code, avoiding ROS imports and node construction."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    original = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    selected = [
        node for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {node.name for node in selected} == set(method_names)
    extracted = ast.ClassDef(
        name=class_name, bases=[], keywords=[], body=selected, decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[class_name]


class FakeInstant:
    nanoseconds = 1_000_000_000

    def __sub__(self, other):
        return SimpleNamespace(nanoseconds=self.nanoseconds - other.nanoseconds)


class FakeLogger:
    def __init__(self):
        self.messages = []

    def __getattr__(self, level):
        def record(message, **kwargs):
            self.messages.append((level, message, kwargs))
        return record


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeTracker:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class FakeModel:
    def __init__(self, tracker):
        self.predictor = SimpleNamespace(trackers=[tracker])
        self.track_calls = 0

    def track(self, _frame, **_kwargs):
        self.track_calls += 1
        boxes = SimpleNamespace(
            id=FakeTensor([101, 102, 103, 104, 105]),
            xyxy=FakeTensor([[x, 20, x + 10, 30] for x in range(20, 120, 20)]),
            conf=FakeTensor([0.9] * 5),
            cls=FakeTensor([0] * 5),
        )
        return [SimpleNamespace(boxes=boxes)]


@pytest.fixture
def harness():
    """Use real method bodies while keeping all hardware interfaces fake."""
    monotonic = SimpleNamespace(value=100)
    cv2 = SimpleNamespace(
        error=ValueError,
        FONT_HERSHEY_SIMPLEX=0,
        putText=lambda *_args, **_kwargs: None,
        rectangle=lambda *_args, **_kwargs: None,
    )
    namespace = {
        'Enum': Enum,
        'np': np,
        'Counter': Counter,
        'cv2': cv2,
        'time': SimpleNamespace(monotonic_ns=lambda: monotonic.value),
    }
    control_path = CONTROL_DIR / '0821auto.py'
    tree = ast.parse(control_path.read_text(encoding='utf-8'))
    enum_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'MissionState'
    )
    exec(compile(ast.Module(body=[enum_node], type_ignores=[]), str(control_path), 'exec'), namespace)
    control_class = _load_class_methods(
        control_path,
        'OffboardControl',
        (
            '_enter_recon_transit', '_start_recon_search',
            '_apply_pending_vision_reset', 'vision_timer_callback',
            'camera_timer_callback',
        ),
        namespace,
    )
    vision_class = _load_class_methods(
        CONTROL_DIR / 'visual_servoing.py',
        'VisualServoingController',
        ('reset_tracking_state', 'process_frame'),
        namespace,
    )
    vision = vision_class()
    vision.tracking_history = deque(maxlen=30)
    vision.tracking_buffer_size = 30
    vision.frame_counter = 37
    vision.is_model_loaded = True
    vision.CONFIDENCE_THRESHOLD = 0.4
    vision.enable_video_recording = False
    vision.enable_photo_capture = False
    vision._pixel_to_world_frd = lambda center, *_args: tuple(map(float, center))
    tracker = FakeTracker()
    vision.model = FakeModel(tracker)
    logger = FakeLogger()
    node = control_class()
    node._vision_result_lock = threading.Lock()
    node._frame_lock = threading.Lock()
    node._map_data_lock = threading.Lock()
    node._vision_ready_event = threading.Event()
    node._vision_ready_event.set()
    node.is_vision_ready = True
    node.vision_controller = vision
    node.vision_epoch = 0
    node.vision_reset_applied_epoch = 0
    node.recon_started_monotonic_ns = None
    node.mission_state = namespace['MissionState'].GLOBAL_SEARCH
    node.current_vision_info = [{'id': 1}]
    node.latest_vision_info = [{'id': 1}]
    node.latest_vision_frame_pose = {'z': -2.0}
    node.latest_vision_frame_sequence = 0
    node.latest_vision_mission_state = node.mission_state
    node.latest_annotated_frame = np.ones((10, 10, 3), dtype=np.uint8)
    node.latest_frame = np.zeros((160, 160, 3), dtype=np.uint8)
    node.latest_frame_pose_snapshot = {'z': -5.0, 'roll': 0.0, 'pitch': 0.0}
    node.latest_frame_sequence = 1
    node.latest_frame_read_started_monotonic_ns = 200
    node.latest_frame_monotonic_ns = 210
    node.latest_frame_received_time_ns = FakeInstant.nanoseconds
    node.last_processed_frame_sequence = -1
    node.initial_z = 0.0
    node.offboard_setpoint_counter = 1
    node.global_map_accepting_samples = False
    node.get_clock = lambda: SimpleNamespace(now=FakeInstant)
    node.get_logger = lambda: logger
    node._capture_widecam_pose_snapshot = lambda: {
        'z': -5.0, 'roll': 0.0, 'pitch': 0.0,
    }
    return SimpleNamespace(
        node=node, vision=vision, tracker=tracker, logger=logger,
        clock=monotonic, states=namespace['MissionState'],
    )


def _seed_old_history(vision):
    for _ in range(30):
        vision.tracking_history.append([
            {'id': index, 'center': (index * 20, 20)} for index in (1, 2, 3)
        ])


def test_first_recon_frame_confirms_all_five_new_ids(harness):
    node, vision = harness.node, harness.vision
    _seed_old_history(vision)
    node._enter_recon_transit()
    assert node.latest_vision_info == []
    assert node.current_vision_info == []
    assert node.latest_vision_frame_pose is None
    assert node.latest_vision_frame_sequence == -1
    assert node.latest_vision_mission_state is None
    assert node.latest_annotated_frame is None
    assert len(vision.tracking_history) == 30  # Reset belongs to the vision callback.
    node._start_recon_search()
    node.vision_timer_callback()
    assert {target['id'] for target in node.latest_vision_info} == set(range(101, 106))
    assert len(vision.tracking_history) == 1
    assert harness.tracker.reset_calls == 1
    assert node.vision_reset_applied_epoch == node.vision_epoch == 1
    assert node.latest_vision_mission_state == harness.states.RECON_SEARCH


def test_repeated_transit_entry_requests_only_one_reset(harness):
    node = harness.node
    node.latest_frame = None
    node._enter_recon_transit()
    node._enter_recon_transit()
    node.vision_timer_callback()
    node._enter_recon_transit()
    node.vision_timer_callback()
    assert node.mission_state == harness.states.RETURN_TO_CENTER_DROPAREA
    assert node.vision_epoch == node.vision_reset_applied_epoch == 1
    assert harness.tracker.reset_calls == 1


def test_repeated_search_entry_preserves_original_cutoff(harness):
    node = harness.node
    node._enter_recon_transit()
    node._start_recon_search()
    harness.clock.value = 500
    node._start_recon_search()
    assert node.recon_started_monotonic_ns == 100
    assert node.mission_state == harness.states.RECON_SEARCH
    assert node.vision_epoch == 1


def test_global_search_keeps_history_and_collects_multiple_samples(harness):
    node, vision = harness.node, harness.vision
    node.global_map_accepting_samples = True
    collected = []
    node._collect_global_search_map_sample = lambda targets, pose: collected.append(
        ([dict(target) for target in targets], dict(pose))
    )
    node.vision_timer_callback()
    node.latest_frame_sequence += 1
    node.vision_timer_callback()
    assert len(collected) == len(vision.tracking_history) == 2
    assert all({target['id'] for target in targets} == {101, 102, 103}
               for targets, _pose in collected)
    assert all(pose['z'] == -5.0 for _targets, pose in collected)
    assert node.latest_vision_mission_state == harness.states.GLOBAL_SEARCH
    assert node.vision_epoch == node.vision_reset_applied_epoch == 0
    assert harness.tracker.reset_calls == 0


@pytest.mark.parametrize('initial_state', ['GLOBAL_SEARCH', 'RECON_SEARCH'])
def test_old_inflight_inference_cannot_publish_after_new_recon_epoch(harness, initial_state):
    node, vision = harness.node, harness.vision
    node.mission_state = getattr(harness.states, initial_state)
    node.recon_started_monotonic_ns = 0
    entered, release = threading.Event(), threading.Event()
    failures = []
    process_frame = vision.process_frame

    def blocked_process(*args, **kwargs):
        entered.set()
        assert release.wait(3), 'test did not release the simulated inference'
        return process_frame(*args, **kwargs)

    def run_callback():
        try:
            node.vision_timer_callback()
        except BaseException as exc:
            failures.append(exc)

    vision.process_frame = blocked_process
    worker = threading.Thread(target=run_callback)
    worker.start()
    try:
        assert entered.wait(3), 'simulated inference did not start'
        node._enter_recon_transit()
        node._start_recon_search()
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert failures == []
    assert node.latest_vision_info == []
    assert node.latest_annotated_frame is None
    assert node.latest_vision_mission_state is None
    assert len(vision.tracking_history) == 0
    assert harness.tracker.reset_calls == 1
    assert node.vision_reset_applied_epoch == node.vision_epoch == 1


def test_inference_exception_still_applies_transition_reset(harness):
    node, vision = harness.node, harness.vision

    def interrupted_process(*_args, **_kwargs):
        node._enter_recon_transit()
        vision.tracking_history.append([{'id': 999, 'center': (10, 10)}])
        raise RuntimeError('simulated inference failure')

    vision.process_frame = interrupted_process
    with pytest.raises(RuntimeError, match='simulated inference failure'):
        node.vision_timer_callback()
    assert len(vision.tracking_history) == 0
    assert harness.tracker.reset_calls == 1
    assert node.vision_reset_applied_epoch == node.vision_epoch == 1
    assert node.latest_vision_info == []


@pytest.mark.parametrize('read_started, should_process',
                         [(None, False), (99, False), (100, True), (101, True)])
def test_recon_rejects_reads_started_before_search_entry(harness, read_started, should_process):
    node = harness.node
    node._enter_recon_transit()
    node._start_recon_search()
    node.latest_frame_read_started_monotonic_ns = read_started
    node.latest_frame_monotonic_ns = 200
    node.vision_timer_callback()
    assert harness.vision.model.track_calls == int(should_process)
    assert bool(node.latest_vision_info) is should_process


def test_read_crossing_search_entry_is_rejected_but_next_read_is_accepted(harness):
    node = harness.node
    node._enter_recon_transit()

    def crossing_read():
        harness.clock.value = 200
        node._start_recon_search()
        harness.clock.value = 250
        return True, np.zeros((160, 160, 3), dtype=np.uint8)

    node.cap = SimpleNamespace(read=crossing_read)
    node.camera_timer_callback()
    assert node.latest_frame_read_started_monotonic_ns == 100
    assert node.latest_frame_monotonic_ns == 250
    assert node.latest_frame_pose_snapshot['frame_read_started_monotonic_ns'] == 100
    assert node.latest_frame_pose_snapshot['frame_received_monotonic_ns'] == 250
    assert node.recon_started_monotonic_ns == 200
    node.vision_timer_callback()
    assert harness.vision.model.track_calls == 0
    harness.clock.value = 300
    node.cap.read = lambda: (True, np.zeros((160, 160, 3), dtype=np.uint8))
    node.camera_timer_callback()
    node.vision_timer_callback()
    assert harness.vision.model.track_calls == 1
    assert len(node.latest_vision_info) == 5


@pytest.mark.parametrize('missing', ['frame', 'pose'])
def test_pending_reset_is_consumed_even_without_valid_image(harness, missing):
    node = harness.node
    _seed_old_history(harness.vision)
    node._enter_recon_transit()
    if missing == 'frame':
        node.latest_frame = None
    else:
        node.latest_frame_pose_snapshot = None
    node.vision_timer_callback()
    assert node.vision_reset_applied_epoch == node.vision_epoch == 1
    assert len(harness.vision.tracking_history) == 0
    assert harness.tracker.reset_calls == 1
    assert harness.vision.model.track_calls == 0


def test_failed_reset_remains_pending_and_prevents_inference_until_retry(harness):
    node, vision = harness.node, harness.vision
    node._enter_recon_transit()
    node._start_recon_search()
    reset_tracking_state = vision.reset_tracking_state
    attempts = []

    def flaky_reset():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError('simulated tracker reset failure')
        reset_tracking_state()

    vision.reset_tracking_state = flaky_reset
    node.vision_timer_callback()
    assert node.vision_reset_applied_epoch == 0
    assert vision.model.track_calls == 0
    errors = [message for message in harness.logger.messages if message[0] == 'error']
    assert errors and errors[-1][2].get('throttle_duration_sec')
    node.vision_timer_callback()
    assert len(attempts) == 2
    assert node.vision_reset_applied_epoch == 1
    assert vision.model.track_calls == 1


@pytest.mark.parametrize('model', [None, SimpleNamespace(), SimpleNamespace(predictor=None),
                                  SimpleNamespace(predictor=SimpleNamespace(trackers=[]))])
def test_reset_handles_absent_model_predictor_or_trackers(harness, model):
    vision = harness.vision
    _seed_old_history(vision)
    vision.model = model
    vision.reset_tracking_state()
    assert len(vision.tracking_history) == 0
    assert vision.tracking_history.maxlen == 30
    assert vision.frame_counter == 37


def test_reset_clears_history_before_propagating_tracker_failure(harness):
    vision = harness.vision
    _seed_old_history(vision)

    def fail():
        raise RuntimeError('tracker failed')

    vision.model.predictor.trackers = [SimpleNamespace(reset=fail)]
    with pytest.raises(RuntimeError, match='tracker failed'):
        vision.reset_tracking_state()
    assert len(vision.tracking_history) == 0


def test_reset_validates_all_tracker_reset_methods_before_invoking_any(harness):
    vision = harness.vision
    tracker = FakeTracker()
    vision.model.predictor.trackers = [tracker, SimpleNamespace()]
    with pytest.raises(RuntimeError, match='reset'):
        vision.reset_tracking_state()
    assert tracker.reset_calls == 0
