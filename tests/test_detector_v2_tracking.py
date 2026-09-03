import numpy as np

from modules.VisionSys.detectorV2 import VisionSystem
from modules.VisionSys.components.objects import ID_Team


def test_predict_ball_uses_roi_converter_when_tracking():
    vs = VisionSystem(debug=False)
    vs.fieldReduce = np.zeros((100, 100, 3), dtype=np.uint8)

    vs.GetRoiImg = lambda ROI_obj, img_shape, min_size=12, max_ratio=0.5: (0, 0, 10, 10)
    vs.ball.get_roi = lambda image_shape, t_now=None, name=None: (0.0, 0.0, 10.0, 10.0)

    roi_img, roi_rect = vs.PredictBall((100, 100), 1.0)

    assert roi_img is not None
    assert roi_rect == (0, 0, 10, 10)


def test_hsv_sampling_uses_small_median_window():
    vs = VisionSystem(debug=False)
    hsv = np.full((5, 5, 3), [0, 0, 0], dtype=np.uint8)
    hsv[1:4, 1:4] = [100, 200, 200]

    sampled = vs._SampleHsvWindow9(hsv, 2, 2, pad=1)
    measured = vs.GetHsvMean(hsv, 2, 2)

    assert np.array_equal(sampled, [100, 200, 200])
    assert np.array_equal(measured, [100, 200, 200])


def test_robot_roi_fallback_confirms_global_candidate_coordinates():
    vs = VisionSystem(debug=False)
    global_hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    vs.imgHSV = global_hsv
    vs.allyColor = np.array([100, 200, 200])

    candidate = {
        "xi": 30,
        "yi": 30,
        "x_m": 30,
        "y_m": 30,
        "xcm": 1.0,
        "ycm": 2.0,
        "direction": np.array([1.0, 0.0]),
        "ri": 8,
        "windowActual": np.zeros((10, 10, 3), dtype=np.uint8),
        "contour": None,
    }
    seen_hsv = []

    vs.DetectPlayerCandidates = lambda *args, **kwargs: ([candidate], [])
    vs.IdentifyCandidateByColor = lambda cand, hsv, main_color: (
        seen_hsv.append(hsv) or
        ({"robot_id": 1, "team": vs.robotAlly1.team}, np.array([1, 2, 3]), np.array([4, 5, 6]))
    )
    vs.GetBotById = lambda team, robot_id: vs.robotAlly1

    results = vs._ProcessROIForRobots(
        np.zeros((10, 10, 3), dtype=np.uint8),
        (20, 20, 10, 10),
        timestamp=1.0,
    )

    assert results[0]["id"] == vs.robotAlly1.id
    assert len(seen_hsv) == 1
    assert seen_hsv[0] is global_hsv


def test_robot_recovery_roi_uses_last_measured_position():
    vs = VisionSystem(debug=False)
    vs.fieldReduce = np.zeros((100, 100, 3), dtype=np.uint8)
    bot = vs.robotAlly1
    bot.position = np.array([10.0, 20.0])
    bot.kalman_initialized = True
    roi_calls = []

    def fake_roi(roi_cm, img_shape):
        roi_calls.append(roi_cm)
        return (20, 30, 40, 40)

    vs.GetRoiImg = fake_roi

    roi_img, roi_rect = vs.PredictRobotRecovery(
        (100, 100), ID_Team.TEAM_ALLY, bot.id
    )

    assert roi_rect == (20, 30, 40, 40)
    assert roi_img.shape == (40, 40, 3)
    assert roi_calls[0] == (-5.0, 5.0, 30.0, 30.0)
