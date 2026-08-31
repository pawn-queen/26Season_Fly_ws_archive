from control.AlignmentChecker import AlignmentChecker


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_checker(clock, logger, time_window=2.0, check_frequency=10):
    return AlignmentChecker(
        logger,
        0.15,
        time_window,
        check_frequency,
        time_func=clock,
    )


def test_alignment_uses_elapsed_time_instead_of_call_count():
    clock = FakeClock()
    messages = []
    checker = make_checker(clock, messages.append, check_frequency=1000)

    assert checker.check(0.0, 0.0, 0.1, 0.0) is False
    for _ in range(100):
        assert checker.check(0.0, 0.0, 0.1, 0.0) is False

    clock.advance(1.99)
    assert checker.check(0.0, 0.0, 0.1, 0.0) is False

    clock.advance(0.01)
    assert checker.check(0.0, 0.0, 0.1, 0.0) is True
    assert len(messages) == 1


def test_out_of_threshold_sample_restarts_continuous_timer():
    clock = FakeClock()
    checker = make_checker(clock, lambda _message: None)

    assert checker.check(0.0, 0.0, 0.1, 0.0) is False
    clock.advance(1.5)
    assert checker.check(0.0, 0.0, 0.15, 0.0) is False

    clock.advance(10.0)
    assert checker.check(0.0, 0.0, 0.1, 0.0) is False
    clock.advance(2.0)
    assert checker.check(0.0, 0.0, 0.1, 0.0) is True


def test_reset_restarts_continuous_timer():
    clock = FakeClock()
    messages = []
    checker = make_checker(clock, messages.append, time_window=1.0)

    assert checker.check(0.0, 0.0, 0.1, 0.0) is False
    clock.advance(1.0)
    assert checker.check(0.0, 0.0, 0.1, 0.0) is True

    checker.reset()
    assert checker.check(0.0, 0.0, 0.1, 0.0) is False
    clock.advance(1.0)
    assert checker.check(0.0, 0.0, 0.1, 0.0) is True
    assert messages[-2] == "对准检查器已重置。"
    assert "误差连续 1.00 秒" in messages[-1]


def test_non_finite_error_never_passes_alignment():
    clock = FakeClock()
    checker = make_checker(clock, lambda _message: None, time_window=0.0)

    assert checker.check(float("nan"), 0.0, 0.0, 0.0) is False
    assert checker.check(0.0, 0.0, 0.0, 0.0) is True


def test_non_finite_clock_never_passes_alignment():
    clock = FakeClock()
    checker = make_checker(clock, lambda _message: None, time_window=0.0)

    clock.now = float('nan')
    assert checker.check(0.0, 0.0, 0.0, 0.0) is False

    clock.now = 1.0
    assert checker.check(0.0, 0.0, 0.0, 0.0) is True


def test_invalid_configuration_is_rejected():
    for threshold, time_window in ((0.0, 1.0), (1.0, -1.0)):
        try:
            AlignmentChecker(
                lambda _message: None,
                threshold=threshold,
                time_window=time_window,
            )
        except ValueError:
            continue
        raise AssertionError("invalid AlignmentChecker configuration accepted")
