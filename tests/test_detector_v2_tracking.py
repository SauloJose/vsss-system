import numpy as np

from modules.VisionSys.detectorV2 import VisionSystem


def test_predict_ball_uses_roi_converter_when_tracking():
    vs = VisionSystem(debug=False)
    vs.fieldReduce = np.zeros((100, 100, 3), dtype=np.uint8)

    vs.GetRoiImg = lambda ROI_obj, img_shape, min_size=12, max_ratio=0.5: (0, 0, 10, 10)
    vs.ball.get_roi = lambda image_shape, t_now=None, name=None: (0.0, 0.0, 10.0, 10.0)

    roi_img, roi_rect = vs.PredictBall((100, 100), 1.0)

    assert roi_img is not None
    assert roi_rect == (0, 0, 10, 10)
