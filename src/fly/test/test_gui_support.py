import subprocess

import pytest

from control import gui_support


def test_missing_display_disables_gui(monkeypatch):
    monkeypatch.setattr(gui_support.os, 'name', 'posix')
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)

    assert not gui_support.opencv_gui_available()


def test_failed_child_probe_disables_gui(monkeypatch):
    monkeypatch.setenv('DISPLAY', ':99')
    monkeypatch.setattr(
        gui_support.subprocess,
        'run',
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )

    assert not gui_support.opencv_gui_available()


def test_successful_child_probe_enables_gui(monkeypatch):
    monkeypatch.setenv('DISPLAY', ':0')
    monkeypatch.setattr(
        gui_support.subprocess,
        'run',
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )

    assert gui_support.opencv_gui_available()


def test_probe_timeout_disables_gui(monkeypatch):
    monkeypatch.setenv('DISPLAY', ':0')

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])

    monkeypatch.setattr(gui_support.subprocess, 'run', raise_timeout)

    assert not gui_support.opencv_gui_available()


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        gui_support.opencv_gui_available(timeout_s=0.0)
