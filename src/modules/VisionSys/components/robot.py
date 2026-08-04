import numpy as np
import cv2
from modules.VisionSys.components.objects import *

class Robot:
    def __init__(self, id: ID_Robots, team: ID_Team, x=0, y=0, r=0,
                 image=cv2.imread('src/data/images/dark_screen.png'),
                 colorTeam=None, colorCar1=None, colorCar2=None,
                 differential_filter=False):
        # Identificadores
        self.id = id
        self.team = team

        # Posição bruta (Visão)
        self.position = np.array([float(x), float(y)])
        self.lastPosition = self.position.copy()
        self.newPosition = self.position.copy()

        # Direção & Ângulo
        self._direction = np.array([1.0, 0.0])
        self._theta = 0.0

        # Geometria
        self.radius = float(r)
        self.objLimit = Circle(Point2D(x, y), self.radius)
        self.bbox = BorderBox(GeometryType.CIRCLE, self.objLimit)
        self.ObjType = ObjTypeMove.MOVING
        self.objTypeSystem = ObjTypeVision.ROBOT

        # Dimensões físicas
        self.axle_length = 2 * self.radius  # cm

        # Status
        self.detected = False
        self.possessionBall = False

        # Coordenadas na imagem
        self.xi, self.yi, self.ri = 0, 0, 0

        # Imagem & Retângulo de Visualização
        self.image = image
        self.dimMatrix = image.shape[1] if image is not None else 0
        self.viewRect = ViewBot(Point2D(self.position[0], self.position[1]), self.dimMatrix)

        # Cores
        self.colorTeam = colorTeam
        self.colorCar1 = colorCar1
        self.colorCar2 = colorCar2

        # Tempo
        self.lastTimestamp = 0
        self.newTimestamp = 0
        self.dT = 0.0

        # EKF
        self.kalman_initialized = False
        self.kalman_last_time = None
        self._init_kalman()

    # --------------------------
    # Inicialização do Kalman
    # --------------------------
    def _init_kalman(self):
        # ESTADO 5D: [x, y, theta, v (linear), omega (angular)]
        self.kalman_state = np.zeros((5, 1), float)
        self.kalman_P = np.eye(5, dtype=float) * 100.0

        # Ruído do Processo (Q): [x, y, theta, v, omega]
        self.kalman_Q = np.diag([0.05, 0.05, 0.01, 10.0, 5.0])

        # Ruído de Medição da Visão (R): Mede [x, y, theta]
        self.kalman_R = np.diag([1.5, 1.5, 0.03])

        self.kalman_initialized = False
        self.kalman_last_time = None

    @staticmethod
    def _normalize_angle(angle):
        """ Normaliza o ângulo para o intervalo [-pi, pi] """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # --------------------------
    # Properties
    # --------------------------
    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, vec):
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return
        self._direction = vec / norm
        self._theta = self._normalize_angle(np.arctan2(self._direction[1], self._direction[0]))

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, ang):
        self._theta = self._normalize_angle(float(ang))
        self._direction = np.array([np.cos(self._theta), np.sin(self._theta)])

    @property
    def position_filtered(self):
        if not self.kalman_initialized:
            return self.position
        return self.kalman_state[:2, 0]

    @property
    def theta_filtered(self):
        if not self.kalman_initialized:
            return self.theta
        return float(self.kalman_state[2, 0])

    @property
    def velocity_filtered(self):
        """ Retorna velocidade linear e angular filtradas: [v, omega] """
        if not self.kalman_initialized:
            return np.array([0.0, 0.0])
        return self.kalman_state[3:5, 0]

    @property
    def omega_filtered(self):
        """ Retorna a velocidade angular filtrada (rad/s) """
        if not self.kalman_initialized:
            return 0.0
        return float(self.kalman_state[4, 0])  # Corrigido para o índice 4 (5D)

    @property
    def wheel_velocities_filtered(self):
        """ Calcula vL e vR a partir de v e omega """
        v, omega = self.velocity_filtered
        vL = v - (omega * self.axle_length / 2.0)
        vR = v + (omega * self.axle_length / 2.0)
        return np.array([vL, vR])

    # --------------------------
    # Atualização de Posição
    # --------------------------
    def setPosition(self, x, y, direction, image, time, wheel_velocities=None):
        self.lastTimestamp = self.newTimestamp
        self.newTimestamp = time
        self.dT = self.newTimestamp - self.lastTimestamp if self.lastTimestamp != 0 else 0.0

        # Atualiza posições brutas
        self.lastPosition = self.position.copy()
        self.position = np.array([x, y], dtype=float)
        self.newPosition = self.position.copy()

        self.direction = direction 
        self.image = image
        if image is not None:
            self.viewRect.setDimension(image.shape[1])

        # Geometria
        self.objLimit = Circle(Point2D(x, y), self.radius)
        self.viewRect.updateViewBot(Point2D(x, y))
        self.updateBbox()

        # Atualiza Kalman com a MEDIÇÃO REAL recebida
        self.update_kalman([x, y, self.theta], time)

    def updatePosition(self, x, y, direction, image, time, wheel_velocities=None):
        self.setPosition(x, y, direction, image, time)

    def setRadius(self, r):
        self.radius = float(r)
        self.axle_length = 2 * self.radius

    def setPositionNoKalman(self, x, y, theta, timestamp, image=None):
        """ Atualiza a posição do robô sem realizar a predição do EKF """
        self.lastPosition = self.position.copy()
        self.position = np.array([x, y], dtype=float)
        self.newPosition = self.position.copy()

        self.theta = theta  # Atualiza direction e normaliza theta automaticamente

        self.lastTimestamp = self.newTimestamp
        self.newTimestamp = timestamp
        self.dT = max(self.newTimestamp - self.lastTimestamp, 1e-3)

        if image is not None:
            self.image = image
            self.viewRect.setDimension(image.shape[1])

        self.objLimit = Circle(Point2D(x, y), self.radius)
        self.updateBbox()
        self.viewRect.updateViewBot(Point2D(x, y))

        self.detected = False

    # --------------------------
    # Modelo Não-Linear do EKF (5D)
    # --------------------------
    def _non_linear_motion_model(self, state, dt):
        x, y, theta, v, omega = state[:, 0]

        x_new = x + v * dt * np.cos(theta)
        y_new = y + v * dt * np.sin(theta)
        theta_new = self._normalize_angle(theta + omega * dt)
        v_new = v
        omega_new = omega

        return np.array([[x_new], [y_new], [theta_new], [v_new], [omega_new]], dtype=float)

    def _jacobian_motion_model(self, state, dt):
        x, y, theta, v, omega = state[:, 0]

        A = np.array([
            [1.0, 0.0, -v * dt * np.sin(theta), dt * np.cos(theta), 0.0],
            [0.0, 1.0,  v * dt * np.cos(theta), dt * np.sin(theta), 0.0],
            [0.0, 0.0, 1.0,                    0.0,                dt],
            [0.0, 0.0, 0.0,                    1.0,                0.0],
            [0.0, 0.0, 0.0,                    0.0,                1.0]
        ], dtype=float)
        return A

    def update_kalman(self, z_list, timestamp):
        x_m, y_m, theta_m = z_list
        theta_m = self._normalize_angle(theta_m)
        z = np.array([[x_m], [y_m], [theta_m]], dtype=float)

        if not self.kalman_initialized:
            self.kalman_state[:3, 0] = [x_m, y_m, theta_m]
            self.kalman_initialized = True
            self.kalman_last_time = timestamp
            return

        dt = max(timestamp - self.kalman_last_time, 1e-3)
        self.kalman_last_time = timestamp

        # 1. PREDIÇÃO
        A = self._jacobian_motion_model(self.kalman_state, dt)
        self.kalman_state = self._non_linear_motion_model(self.kalman_state, dt)
        self.kalman_P = A @ self.kalman_P @ A.T + self.kalman_Q

        # 2. CORREÇÃO
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=float)

        pred = H @ self.kalman_state
        y_residual = z - pred
        y_residual[2, 0] = self._normalize_angle(y_residual[2, 0])

        S = H @ self.kalman_P @ H.T + self.kalman_R
        K = self.kalman_P @ H.T @ np.linalg.inv(S)

        self.kalman_state = self.kalman_state + K @ y_residual
        self.kalman_state[2, 0] = self._normalize_angle(self.kalman_state[2, 0])

        # Forma de Joseph para estabilidade numérica de P
        I_KH = np.eye(5, dtype=float) - K @ H
        self.kalman_P = I_KH @ self.kalman_P @ I_KH.T + K @ self.kalman_R @ K.T

    # --------------------------
    # ROI & Predições Futuras
    # --------------------------
    def get_roi(self, image_shape, t_now, vision_sys, scale_std=3):
        if not self.kalman_initialized or self.kalman_last_time is None:
            return 0, 0, image_shape[1], image_shape[0]

        st_pred, P_pred = self.predict_with_cov(t_now)
        x_pred = st_pred[0, 0]
        y_pred = st_pred[1, 0]

        x_img, y_img = vision_sys.getImageRealIndice([x_pred, y_pred])

        if vision_sys.viewCapture.cooVetor is not None:
            x_offset, y_offset = vision_sys.viewCapture.cooVetor[0], vision_sys.viewCapture.cooVetor[1]
            x_c = x_img - x_offset
            y_c = y_img - y_offset
        else:
            x_c, y_c = x_img, y_img

        pixels_per_cm = (self.ri / self.radius) if self.ri > 0 else 5.0

        std_x = np.sqrt(P_pred[0, 0]) * pixels_per_cm
        std_y = np.sqrt(P_pred[1, 1]) * pixels_per_cm

        w_roi = int(scale_std * std_x * 2)
        h_roi = int(scale_std * std_y * 2)

        min_dimension = int(3.0 * self.ri) if self.ri > 0 else int(2.5 * self.radius * pixels_per_cm)
        w_roi = max(w_roi, min_dimension)
        h_roi = max(h_roi, min_dimension)

        x = int(x_c - w_roi // 2)
        y = int(y_c - h_roi // 2)

        h_img, w_img = image_shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w_roi = min(w_roi, w_img - x)
        h_roi = min(h_roi, h_img - y)

        return x, y, w_roi, h_roi

    def predict(self, time):
        """ Prediz o estado futuro [x, y, theta] usando modelo não-linear 5D """
        if not self.kalman_initialized:
            return self.position[0], self.position[1], self.theta

        dt = max(time - self.kalman_last_time, 0.0)
        st_predicted = self._non_linear_motion_model(self.kalman_state, dt)

        return st_predicted[0, 0], st_predicted[1, 0], st_predicted[2, 0]

    def predict_with_cov(self, time):
        """ Retorna (state_pred, P_pred) em 5D sem alterar estado interno """
        if not self.kalman_initialized:
            st = np.zeros((5, 1), dtype=float)
            st[:3, 0] = [self.position[0], self.position[1], self.theta]
            P_temp = np.eye(5, dtype=float) * 50.0
            return st, P_temp

        dt = max(time - self.kalman_last_time, 0.0)

        st_pred = self._non_linear_motion_model(self.kalman_state, dt)
        A = self._jacobian_motion_model(st_pred, dt)
        P_pred = A @ self.kalman_P @ A.T + self.kalman_Q

        st_pred[2, 0] = self._normalize_angle(st_pred[2, 0])
        return st_pred, P_pred

    # --------------------------
    # Métodos Auxiliares
    # --------------------------
    def updateBbox(self):
        self.objLimit = Circle(Point2D(self.position[0], self.position[1]), self.radius)
        self.bbox.attPosition(self.objLimit)

    def updtPositionImg(self, xi, yi, ri):
        self.xi = xi
        self.yi = yi
        self.ri = ri

    def getAngle(self):
        return float(self.theta)

    def getColors(self):
        return self.colorTeam, self.colorCar1, self.colorCar2

    def getStatus(self):
        return self.detected

    def setStatus(self, status):
        self.detected = status

    def setColor(self, colorT=None, colorP=None, colorS=None):
        self.colorTeam = colorT
        self.colorCar1 = colorP
        self.colorCar2 = colorS

    def setColorTeam(self, colorT=None):
        self.colorTeam = colorT

    def setDirection(self, direction):
        self.direction = direction

    def setTeamColor(self, color):
        self.teamColor = color

    def reset(self):
        self.position = np.array([0.0, 0.0])
        self.lastPosition = self.position.copy()
        self.newPosition = self.position.copy()
        self.direction = np.array([1.0, 0.0])
        self.theta = 0.0
        self.detected = False
        self.possessionBall = False
        self.dT = 0.0
        self.lastTimestamp = 0
        self.newTimestamp = 0
        self._init_kalman()

    def resetState(self):
        self.reset()