import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'src'))


@pytest.fixture
def vision_system(monkeypatch):
    # Stub de dependências pesadas para importar o módulo sem abrir UI
    monkeypatch.setitem(sys.modules, 'tkinter', types.SimpleNamespace())
    from modules.VisionSys.detectorV2 import VisionSystem

    vs = VisionSystem(debug=False)
    vs.frameResult = np.zeros((64, 64, 3), dtype=np.uint8)
    return vs


def test_processimg_returns_frame_when_debug_enabled(vision_system):
    img = np.zeros((64, 64, 3), dtype=np.uint8)

    vision_system.Proc = lambda *args, **kwargs: img.copy()
    vision_system.DrawAllRobots = lambda: None

    result = vision_system.ProcessImg(img, debug=True)

    assert result is not None
    assert getattr(result, 'shape', None) == (64, 64, 3)


def test_processimg_uses_proc_path_for_image_mode(vision_system):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    calls = []

    def fake_proc(*args, **kwargs):
        calls.append(('proc', args[0].shape))
        return img.copy()

    vision_system.emulatorMode = 'image'
    vision_system.Proc = fake_proc
    vision_system.DrawAllRobots = lambda: None

    result = vision_system.ProcessImg(img, debug=False)

    assert result is not None
    assert calls[0][0] == 'proc'


def test_pipeline_calls_filtered_detection_in_video_mode(monkeypatch, vision_system):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    calls = []

    def fake_filtered(*args, **kwargs):
        calls.append('filtered')
        vision_system.frameResult = img.copy()

    vision_system.emulatorMode = 'video'
    vision_system.fieldDetectedFlag = True
    vision_system._count = 12000
    vision_system.lastMajorTime = 0
    vision_system.newProcTime = 1.0
    vision_system.FilteredDetection = fake_filtered
    vision_system.DrawAllRobots = lambda: None

    result = vision_system.ProcessImg(img, debug=False)

    assert result is not None
    assert calls == ['filtered']
