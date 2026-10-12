import numpy as np

from modules.VisionSys.components.ball import Ball
from modules.VisionSys.components.objects import ID_Robots, ID_Team
from modules.VisionSys.components.robot import Robot


def test_ball_kalman_updates_position_and_predicts_without_mutating_state():
    ball = Ball()
    ball.setPosition(10.0, 20.0, 2.0, timestamp=1.0)
    ball.update_kalman([12.0, 20.0], timestamp=2.0)

    filtered_position = ball.position_filtered.copy()
    filtered_velocity = ball.velocity_filtered.copy()
    state_before_prediction = ball.kalman_state.copy()
    predicted_x, predicted_y, _ = ball.predict(timestamp=3.0)

    assert ball.kalman_state.shape == (4, 1)
    assert np.all(np.isfinite(ball.kalman_state))
    assert 10.0 < filtered_position[0] < 12.0
    assert filtered_position[1] == 20.0
    assert filtered_velocity[0] > 0.0
    assert predicted_x > filtered_position[0]
    assert predicted_y == filtered_position[1]
    np.testing.assert_array_equal(ball.kalman_state, state_before_prediction)


def test_robot_ekf_normalizes_angle_residual_across_pi_boundary():
    robot = Robot(
        ID_Robots.ROBOT_ALLY_GOAL,
        ID_Team.TEAM_ALLY,
        image=None,
    )
    robot.update_kalman([10.0, 20.0, np.pi - 0.05, 0.0, 0.0], timestamp=1.0)
    robot.update_kalman([10.0, 20.0, -np.pi + 0.05, 0.0, 0.0], timestamp=2.0)

    angle = robot.theta_filtered
    angular_error = abs((angle - np.pi + np.pi) % (2.0 * np.pi) - np.pi)

    assert robot.kalman_state.shape == (5, 1)
    assert np.all(np.isfinite(robot.kalman_state))
    assert angular_error < 0.1


def test_robot_prediction_propagates_motion_without_mutating_filter():
    robot = Robot(
        ID_Robots.ROBOT_ALLY_GOAL,
        ID_Team.TEAM_ALLY,
        image=None,
    )
    robot.update_kalman([0.0, 0.0, 0.0, 1.0, 0.0], timestamp=1.0)
    state_before_prediction = robot.kalman_state.copy()

    x, y, theta = robot.predict(time=2.0)
    predicted_state, predicted_covariance = robot.predict_with_cov(time=2.0)

    assert x == 1.0
    assert y == 0.0
    assert theta == 0.0
    assert predicted_state.shape == (5, 1)
    assert predicted_covariance.shape == (5, 5)
    np.testing.assert_array_equal(robot.kalman_state, state_before_prediction)
