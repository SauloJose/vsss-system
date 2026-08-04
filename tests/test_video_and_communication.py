import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'src'))


VIDEO_DIR = Path(ROOT) / 'src' / 'data' / 'videos'
VIDEO_FILES = sorted(VIDEO_DIR.glob('*.mp4'))


@pytest.fixture
def cv2_available():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not VIDEO_FILES, reason='videos not found')
def test_video_files_exist_and_are_readable():
    assert VIDEO_FILES, 'expected video files under src/data/videos'
    for video in VIDEO_FILES:
        assert video.exists()
        assert video.stat().st_size > 0


@pytest.mark.skipif(not VIDEO_FILES, reason='videos not found')
def test_open_video_with_opencv(cv2_available):
    if not cv2_available:
        pytest.skip('opencv not available')

    import cv2

    video_path = str(VIDEO_FILES[0])
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'could not open video {video_path}'

    ok, frame = cap.read()
    assert ok, 'could not read first frame from video'
    assert frame is not None
    assert frame.ndim == 3

    cap.release()


def test_states_transmitter_builds_frame_payload():
    from modules.control.comm.public.state_tx import StatesTransmissor

    class DummyFieldData:
        def __init__(self):
            class DummyPosition:
                x = 1.0
                y = 2.0
                theta = 0.5
            class DummyVelocity:
                x = 0.1
                y = 0.2
                theta = 0.3
            self.ball = types.SimpleNamespace(position=DummyPosition(), velocity=DummyVelocity())
            self.robots = [types.SimpleNamespace(position=DummyPosition(), velocity=DummyVelocity()) for _ in range(3)]
            self.foes = [types.SimpleNamespace(position=DummyPosition(), velocity=DummyVelocity()) for _ in range(3)]

    tx = StatesTransmissor(ip='127.0.0.1', port=10002)
    tx.transmit = lambda frame: setattr(tx, '_last_frame', frame)

    field_data = DummyFieldData()
    tx.send_state(field_data)

    frame = tx._last_frame
    assert frame.ball.x == 1.0
    assert len(frame.robots_blue) == 3
    assert len(frame.robots_yellow) == 3


def test_commands_receiver_parses_message():
    from modules.control.comm.public.commands_rx import CommandsReceiver
    from modules.control.comm.protocols import command_pb2

    class DummyReceiver:
        def __init__(self, ip='224.0.0.1', port=10003):
            self.ip = ip
            self.port = port

        def receive(self):
            pb = command_pb2.Commands()
            cmd = pb.robot_commands.add()
            cmd.id = 1
            cmd.yellowteam = False
            cmd.wheel_left = 3.0
            cmd.wheel_right = 4.0
            return pb.SerializeToString()

    with patch('modules.control.comm.public.commands_rx.Receiver', DummyReceiver):
        receiver = CommandsReceiver(ip='127.0.0.1', port=10003)
        team_cmd = receiver.receive_and_convert()

    assert team_cmd is not None
    assert team_cmd.commands[1].left_speed == 3.0
    assert team_cmd.commands[1].right_speed == 4.0


def test_emulator_initialization_smoke(monkeypatch):
    import modules.emulator.emulator as emulator_module

    class DummyApp:
        menu = types.SimpleNamespace()
        viewer = types.SimpleNamespace()
        debugField = types.SimpleNamespace()
        debugObject = types.SimpleNamespace()
        debugPlayers = types.SimpleNamespace()
        debugTeam = types.SimpleNamespace()
        result = types.SimpleNamespace()
        virtualVision = types.SimpleNamespace()
        cards = []
        infosEmulator = types.SimpleNamespace()
        IdFrame = 2
        btn_run = types.SimpleNamespace()
        btn_pause = types.SimpleNamespace()
        btn_stop = types.SimpleNamespace()

    monkeypatch.setattr(emulator_module, 'VisionSystem', lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(emulator_module, 'MyViewer', lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(emulator_module, 'WindowsViewer', lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(emulator_module, 'CardInfos', lambda *a, **k: types.SimpleNamespace())

    app = DummyApp()
    emu = emulator_module.Emulator(app)

    assert emu is not None
