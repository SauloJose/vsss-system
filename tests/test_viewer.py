import types

import numpy as np


def test_viewer_converts_opencv_frame_with_pillow(monkeypatch):
    from ui import viewer

    displayed = {}
    monkeypatch.setattr(viewer.ImageTk, "PhotoImage", lambda image: image)

    instance = viewer.MyViewer.__new__(viewer.MyViewer)
    instance.canvas = types.SimpleNamespace(
        create_image=lambda *args, **kwargs: displayed.update(kwargs)
    )

    instance.show(np.zeros((20, 40, 3), dtype=np.uint8))

    assert isinstance(instance.image, viewer.PILImage.Image)
    assert instance.image.size == (748, 374)
    assert displayed["image"] is instance.image


def test_viewer_skips_resize_when_frame_already_fits(monkeypatch):
    from ui import viewer

    monkeypatch.setattr(viewer.ImageTk, "PhotoImage", lambda image: image)

    def unexpected_resize(*args, **kwargs):
        raise AssertionError("resize should be skipped when dimensions are unchanged")

    monkeypatch.setattr(viewer.PILImage.Image, "resize", unexpected_resize)
    instance = viewer.MyViewer.__new__(viewer.MyViewer)
    instance.canvas = types.SimpleNamespace(create_image=lambda *args, **kwargs: None)

    instance.show(np.zeros((480, 640, 3), dtype=np.uint8))

    assert instance.image.size == (640, 480)
