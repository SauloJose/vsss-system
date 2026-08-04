import os
import sys
import types
from unittest.mock import patch

import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'src'))


class DummyRoot:
    def __init__(self):
        self.children = []

    def title(self, *args, **kwargs):
        return None

    def configure(self, *args, **kwargs):
        return None

    def geometry(self, *args, **kwargs):
        return None

    def resizable(self, *args, **kwargs):
        return None

    def iconbitmap(self, *args, **kwargs):
        return None

    def mainloop(self, *args, **kwargs):
        return None


@pytest.fixture
def fake_tk(monkeypatch):
    fake_tk_module = types.SimpleNamespace(
        Tk=lambda: DummyRoot(),
        Frame=lambda *args, **kwargs: types.SimpleNamespace(place=lambda *a, **k: None, pack=lambda *a, **k: None),
        Label=lambda *args, **kwargs: types.SimpleNamespace(place=lambda *a, **k: None, pack=lambda *a, **k: None, config=lambda *a, **k: None),
        Button=lambda *args, **kwargs: types.SimpleNamespace(place=lambda *a, **k: None, pack=lambda *a, **k: None, config=lambda *a, **k: None),
        BOTH='both',
        X='x',
        LEFT='left',
        RIGHT='right',
        TOP='top',
        BOTTOM='bottom',
        Y='y',
        N='n',
        S='s',
        W='w',
        E='e',
        NORMAL='normal',
        DISABLED='disabled',
    )
    monkeypatch.setitem(sys.modules, 'tkinter', fake_tk_module)
    monkeypatch.setitem(sys.modules, 'tkinter.ttk', types.SimpleNamespace(Notebook=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, 'ttkthemes', types.SimpleNamespace(ThemedStyle=lambda *a, **k: None))
    return fake_tk_module


def test_parse_requirements_reads_file(tmp_path):
    from main import parse_requirements

    req_file = tmp_path / 'requirements.txt'
    req_file.write_text('numpy==1.24\n# comment\nopencv-python\n', encoding='utf-8')

    packages = parse_requirements(str(req_file))

    assert packages == ['numpy', 'opencv-python']


def test_check_missing_packages_returns_missing(fake_tk):
    import main

    with patch.object(main, 'parse_requirements', return_value=['numpy', 'opencv-python']):
        with patch.object(main, 'is_installed', side_effect=[False, True]):
            missing = main.check_missing_packages()

    assert missing == ['numpy']
