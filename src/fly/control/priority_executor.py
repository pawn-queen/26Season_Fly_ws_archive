"""Five callback workers with separate flight and auxiliary scheduling lanes.

This is CPU resource isolation, not a realtime or preemptive executor.  The
flight lane has three workers; camera, vision, preview and unregistered groups
share two workers whose Linux nice value is increased by five.  The executor
uses public rclpy APIs and supports the synchronous callbacks used by control.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import inspect
import os
import threading
import time
import warnings

from rclpy.executors import (
    ConditionReachedException,
    Executor,
    ExternalShutdownException,
    ShutdownException,
    TimeoutException,
)


@dataclass
class _GroupLane:
    group: object
    critical: bool
    entities: set = field(default_factory=set)
    pending: deque = field(default_factory=deque)
    queued_entities: dict = field(default_factory=dict)
    running: object = None


@dataclass
class _Work:
    handler: object
    entity: object
    lane: _GroupLane


class PriorityExecutor(Executor):
    """Keep auxiliary callbacks off the three reserved flight workers.

    ``critical_groups`` gives the initial round-robin order.  Other groups are
    auxiliary unless explicitly registered or owned by ``add_critical_node``.
    Every group remains serial inside this executor, including default groups.
    ROS messages are taken by rclpy when their handler actually runs; this
    scheduler does not copy messages or replay missed timer periods.

    Callback exceptions are re-raised by a subsequent ``spin_once`` so the
    existing control spin loop can log them and continue.  Shutdown cancels
    work which has not started and joins running workers before returning True.
    A timed-out shutdown returns False and leaves nodes/resources intact.
    """

    CRITICAL_WORKERS = 3
    AUXILIARY_WORKERS = 2
    AUXILIARY_NICE_INCREMENT = 5
    GENERIC_TASK_CAPACITY = 16

    def __init__(
        self,
        *,
        critical_groups=(),
        auxiliary_groups=(),
        context=None,
        priority_warning=None,
    ):
        super().__init__(context=context)
        self._condition = threading.Condition(threading.RLock())
        self._dispatch_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._groups = {}
        self._round_robin = {True: deque(), False: deque()}
        self._running_counts = {True: 0, False: 0}
        self._known_handlers = set()
        self._generic_tasks = set()
        self._generic_group = object()
        self._critical_nodes = set()
        self._errors = deque()
        self._worker_priorities = {}
        self._priority_warning = priority_warning
        self._nice_warning_reported = False
        self._pools = {
            True: ThreadPoolExecutor(
                max_workers=self.CRITICAL_WORKERS,
                thread_name_prefix="offboard-critical",
                initializer=self._initialize_worker,
                initargs=(True,),
            ),
            False: ThreadPoolExecutor(
                max_workers=self.AUXILIARY_WORKERS,
                thread_name_prefix="offboard-auxiliary",
                initializer=self._initialize_worker,
                initargs=(False,),
            ),
        }
        for group in critical_groups:
            self.register_callback_group(group, critical=True)
        for group in auxiliary_groups:
            self.register_callback_group(group, critical=False)

    def register_callback_group(self, group, *, critical):
        """Register a group before spinning; reclassification while busy fails."""
        with self._condition:
            if self._closing:
                raise ShutdownException()
            lane = self._groups.get(group)
            if lane is None:
                lane = _GroupLane(group=group, critical=bool(critical))
                self._groups[group] = lane
                self._round_robin[lane.critical].append(lane)
            elif lane.critical != bool(critical):
                if lane.pending or lane.running is not None:
                    raise RuntimeError("Cannot reclassify an active callback group")
                self._round_robin[lane.critical].remove(lane)
                lane.critical = bool(critical)
                self._round_robin[lane.critical].append(lane)
        self.wake()

    def _register_node_entities(self, node):
        # These are public Node iterators.  There is no copied ROS wait-set or
        # manipulation of entity._executor_event / Executor._make_handler.
        for name in ("subscriptions", "timers", "services", "clients", "guards", "waitables"):
            for entity in getattr(node, name):
                group = entity.callback_group
                with self._condition:
                    if group not in self._groups:
                        self.register_callback_group(
                            group, critical=node in self._critical_nodes
                        )
                    self._groups[group].entities.add(entity)

    def add_node(self, node):
        self._register_node_entities(node)
        return super().add_node(node)

    def add_critical_node(self, node):
        """Add ServoControl (including its default group) to the flight lane."""
        with self._condition:
            self._critical_nodes.add(node)
        for name in ("subscriptions", "timers", "services", "clients", "guards", "waitables"):
            for entity in getattr(node, name):
                self.register_callback_group(entity.callback_group, critical=True)
        return self.add_node(node)

    def create_task(self, callback, *args, **kwargs):
        """Bound non-entity work as well; only synchronous tasks are supported."""
        if inspect.iscoroutinefunction(callback) or inspect.iscoroutine(callback):
            raise TypeError("PriorityExecutor requires synchronous callbacks/tasks")
        with self._condition:
            if self._closing:
                raise ShutdownException()
            if len(self._generic_tasks) >= self.GENERIC_TASK_CAPACITY:
                raise RuntimeError("PriorityExecutor non-entity task capacity exceeded")
            task = super().create_task(callback, *args, **kwargs)
            self._generic_tasks.add(task)
            return task

    def can_execute(self, entity):
        """Do not collect more work for a running group or a queued entity."""
        with self._condition:
            if self._closing:
                return False
            lane = self._groups.get(entity.callback_group)
            if lane is not None and (
                lane.running is not None or entity in lane.queued_entities
            ):
                return False
        return super().can_execute(entity)

    def _enqueue_ready(self, handler, entity, node):
        with self._condition:
            if handler in self._known_handlers:
                return
            if handler.done() or handler.cancelled():
                self._generic_tasks.discard(handler)
                return
            if self._closing:
                handler.cancel()
                self._generic_tasks.discard(handler)
                return
            if entity is not None and inspect.iscoroutinefunction(
                getattr(entity, "callback", None)
            ):
                # Reject before running: a suspended ROS Task may already
                # schedule its own resume inside rclpy, which is outside this
                # executor's synchronous group/accounting contract.
                handler.cancel()
                self._errors.append(TypeError(
                    "PriorityExecutor requires synchronous ROS callbacks"
                ))
                self.wake()
                return
            if entity is None:
                if self._generic_group not in self._groups:
                    self.register_callback_group(self._generic_group, critical=False)
                lane = self._groups[self._generic_group]
            else:
                group = entity.callback_group
                if group not in self._groups:
                    self.register_callback_group(
                        group, critical=node in self._critical_nodes
                    )
                lane = self._groups[group]
                # Handle entities added through normal public Node APIs after
                # add_node.  Capacity is still one queued handler per entity.
                lane.entities.add(entity)
                if entity in lane.queued_entities:
                    # Humble retains each entity until its handler takes it.
                    # A second *different* handler here violates that contract;
                    # retaining the first silently would lose ROS work.
                    raise RuntimeError("ROS entity returned multiple pending handlers")
                lane.queued_entities[entity] = handler
            self._known_handlers.add(handler)
            lane.pending.append(_Work(handler, entity, lane))

    def _dispatch_ready(self):
        with self._condition:
            if self._closing:
                return
            for critical, limit in (
                (True, self.CRITICAL_WORKERS),
                (False, self.AUXILIARY_WORKERS),
            ):
                order = self._round_robin[critical]
                attempts = len(order)
                while attempts and self._running_counts[critical] < limit:
                    lane = order.popleft()
                    order.append(lane)
                    attempts -= 1
                    if lane.running is not None or not lane.pending:
                        continue
                    item = lane.pending.popleft()
                    if item.entity is not None:
                        del lane.queued_entities[item.entity]
                    lane.running = item
                    self._running_counts[critical] += 1
                    # Submit only work backed by an available worker.  Pending
                    # ROS callbacks stay in our bounded per-entity queues.
                    self._pools[critical].submit(self._run_work, item)

    def _run_work(self, item):
        error = None
        try:
            item.handler()
            if not item.handler.done() and not item.handler.cancelled():
                # Closing a suspended handler also unwinds rclpy's group and
                # work-tracker context managers.  Leaving it suspended would
                # retain the group forever and prevent orderly shutdown.
                item.handler.cancel()
                raise RuntimeError(
                    "An asynchronous callback suspended in PriorityExecutor; "
                    "control callbacks must remain synchronous"
                )
            if item.handler.done():
                item.handler.result()
        except BaseException as exc:
            error = exc
        finally:
            with self._condition:
                if error is not None:
                    self._errors.append(error)
                item.lane.running = None
                self._running_counts[item.lane.critical] -= 1
                self._known_handlers.discard(item.handler)
                self._generic_tasks.discard(item.handler)
                self._condition.notify_all()
            self.wake()

    def _raise_callback_error(self):
        with self._condition:
            error = self._errors.popleft() if self._errors else None
        if error is not None:
            raise error

    def _scheduler_needs_attention(self):
        # A worker completion wakes rclpy's wait-set.  The public condition
        # argument makes that wake return to our scheduler rather than sleep
        # again until timeout while already-collected callbacks wait in queues.
        with self._condition:
            if self._closing or self._errors:
                return True
            return any(
                self._running_counts[critical] < limit
                and any(lane.pending and lane.running is None for lane in order)
                for critical, limit, order in (
                    (True, self.CRITICAL_WORKERS, self._round_robin[True]),
                    (False, self.AUXILIARY_WORKERS, self._round_robin[False]),
                )
            )

    def _stop_or_error(self):
        with self._condition:
            return self._closing or bool(self._errors)

    def spin_once(self, timeout_sec=None):
        """Collect a bounded ready batch, then fairly fill the two lanes."""
        with self._dispatch_lock:
            if self._closing:
                raise ShutdownException()
            self._dispatch_ready()
            self._raise_callback_error()
            # Include non-entity tasks, and do not let continuously-ready ROS
            # traffic monopolize the dispatcher.  At most one pending handler
            # per registered entity can be collected in a ready batch.
            with self._condition:
                budget = max(
                    1,
                    sum(len(lane.entities) for lane in self._groups.values())
                    + self.GENERIC_TASK_CAPACITY,
                )
            for index in range(budget):
                if self._closing:
                    break
                try:
                    handler, entity, node = self.wait_for_ready_callbacks(
                        timeout_sec=timeout_sec if index == 0 else 0.0,
                        condition=(
                            self._scheduler_needs_attention if index == 0
                            else self._stop_or_error
                        ),
                    )
                except (TimeoutException, ConditionReachedException):
                    break
                self._enqueue_ready(handler, entity, node)
            self._dispatch_ready()
            self._raise_callback_error()

    def _initialize_worker(self, critical):
        tid = threading.get_native_id()
        baseline = requested = actual = None
        problem = None
        try:
            baseline = os.getpriority(os.PRIO_PROCESS, tid)
            requested = baseline if critical else min(
                19, baseline + self.AUXILIARY_NICE_INCREMENT
            )
            if not critical:
                if requested == baseline:
                    raise RuntimeError("Auxiliary thread nice is already 19")
                os.setpriority(os.PRIO_PROCESS, tid, requested)
            actual = os.getpriority(os.PRIO_PROCESS, tid)
        except (AttributeError, OSError, RuntimeError) as exc:
            problem = str(exc)
            try:
                actual = os.getpriority(os.PRIO_PROCESS, tid)
            except (AttributeError, OSError):
                pass
        with self._condition:
            self._worker_priorities[tid] = {
                "critical": critical,
                "baseline_nice": baseline,
                "requested_nice": requested,
                "actual_nice": actual,
                "error": problem,
            }
            report = (
                not critical and problem is not None
                and not self._nice_warning_reported
            )
            if report:
                self._nice_warning_reported = True
        if report:
            message = (
                "辅助回调线程 nice 降优先级失败；保留 3+2 工作线程隔离，"
                f"未获得额外 CPU 调度降优先级：{problem}"
            )
            try:
                if self._priority_warning is not None:
                    self._priority_warning(message)
                else:
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
            except Exception:
                # Logging must not prevent a thread-pool worker from starting.
                try:
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                except Exception:
                    pass

    def diagnostics_snapshot(self):
        """Return plain diagnostic data without publishing a ROS interface."""
        with self._condition:
            groups = [
                {
                    "critical": lane.critical,
                    "pending": len(lane.pending),
                    "running": lane.running is not None,
                    "entity_capacity": len(lane.entities),
                    "pending_capacity": (
                        self.GENERIC_TASK_CAPACITY
                        if lane.group is self._generic_group else len(lane.entities)
                    ),
                    "generic": lane.group is self._generic_group,
                }
                for lane in self._groups.values()
            ]
            return {
                "critical_running": self._running_counts[True],
                "auxiliary_running": self._running_counts[False],
                "pending": sum(group["pending"] for group in groups),
                "groups": groups,
                "worker_priorities": dict(self._worker_priorities),
                "closing": self._closing,
                "closed": self._closed,
            }

    def shutdown(self, timeout_sec=None):
        """Stop admission, cancel queued tasks, and join before node teardown."""
        deadline = None if timeout_sec is None or timeout_sec < 0 else (
            time.monotonic() + timeout_sec
        )
        with self._close_lock:
            if self._closed:
                return True
            with self._condition:
                self._closing = True
            self.wake()
            # The control wrapper stops/joins its dispatcher before this call.
            # This lock additionally makes standalone use safe against a last
            # spin_once collecting a ready handler while shutdown starts.
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            acquired = self._dispatch_lock.acquire(
                timeout=-1 if remaining is None else remaining
            )
            if not acquired:
                return False
            try:
                with self._condition:
                    for lane in self._groups.values():
                        while lane.pending:
                            item = lane.pending.popleft()
                            item.handler.cancel()
                            self._known_handlers.discard(item.handler)
                            self._generic_tasks.discard(item.handler)
                        lane.queued_entities.clear()
                    for task in tuple(self._generic_tasks):
                        if task not in self._known_handlers:
                            task.cancel()
                            self._generic_tasks.discard(task)
                    while any(self._running_counts.values()):
                        remaining = None if deadline is None else deadline - time.monotonic()
                        if remaining is not None and remaining <= 0:
                            return False
                        self._condition.wait(remaining)
                for pool in self._pools.values():
                    pool.shutdown(wait=True)
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                complete = super().shutdown(timeout_sec=remaining)
                if complete:
                    self._closed = True
                return complete
            finally:
                self._dispatch_lock.release()
