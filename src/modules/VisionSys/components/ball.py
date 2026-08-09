import numpy as np
from modules.VisionSys.components.objects import *

class Ball:
    """
    Classe da bola com filtro de Kalman (estado: x, y, vx, vy).
    A orientação (theta) é derivada da velocidade filtrada.
    """

    def __init__(self, x=0, y=0, r=0):
        # --- Estado geométrico ---
        self.position = np.array([x, y], dtype=float)
        self.radius = float(r)

        # Direção (unitária) e theta serão derivados da velocidade
        self._direction = np.array([0.0, 0.0], dtype=float)
        self._theta = 0.0

        # --- Estruturas geométricas ---
        self.objLimit = Circle(Point2D(x, y), self.radius)
        self.bbox = BorderBox(GeometryType.CIRCLE, self.objLimit)
        self.viewBall = ViewBot(Point2D(x, y), int(r + 14))

        # --- Status ---
        self.ObjType = ObjTypeMove.MOVING
        self.objTypeSystem = ObjTypeVision.BALL
        self.status = False

        # --- Tempo ---
        self.oldTimestamp = 0.0
        self.newTimestamp = 0.0
        self.dT = 0.0

        # --- Movimento ---
        self.lastPosition = self.position.copy()
        self.newPosition = self.position.copy()
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.omega = 0.0  # Velocidade angular (não usada, mantida para compatibilidade)

        # --- Kalman: estado [x, y, vx, vy] (4D) ---
        self.kalman_initialized = False
        self.kalman_last_time = None

        self.kalman_state = np.zeros((4, 1))
        self.kalman_P = np.eye(4) * 100.0          # incerteza inicial

        self.kalman_Q = np.diag([1.0, 1.0, 10.0, 10.0])   # ruído do processo
        self.kalman_R = np.diag([3.0, 3.0])               # ruído da medição (x, y)

        # Cor da bola (médio HSV)
        self.color = None

    # ============================================================
    #         PROPRIEDADES DE DIREÇÃO (derivadas da velocidade)
    # ============================================================

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, d):
        # Mantido para compatibilidade, mas prefira derivar da velocidade
        d = np.asarray(d, dtype=float)
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            return
        d = d / norm
        self._direction = d
        self._theta = float(np.arctan2(d[1], d[0]))

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, ang):
        self._theta = float(ang)
        self._direction = np.array([np.cos(self._theta),
                                    np.sin(self._theta)], dtype=float)

    def _update_direction_from_velocity(self):
        """Atualiza direção e theta a partir da velocidade filtrada (ou atual)."""
        vx, vy = self.velocity_filtered if self.kalman_initialized else self.velocity
        norm = np.hypot(vx, vy)
        if norm > 1e-6:
            self._direction = np.array([vx / norm, vy / norm])
            self._theta = float(np.arctan2(vy, vx))
        # se velocidade zero, mantém a direção anterior

    # ======================================================================
    # 🔹 Atualização de posição
    # ======================================================================

    def setPosition(self, x, y, r, timestamp=0.0, theta=None):
        """
        Define posição inicial da bola (primeira medição).
        O theta é ignorado – será derivado da velocidade (zero inicialmente).
        """
        self.radius = float(r)

        self.position = np.array([x, y], dtype=float)
        self.lastPosition = self.position.copy()
        self.newPosition = self.position.copy()

        # Velocidade inicial nula
        self.velocity = np.array([0.0, 0.0])
        self._update_direction_from_velocity()

        self.oldTimestamp = timestamp
        self.newTimestamp = timestamp
        self.dT = 0.0

        self.updateBbox()
        self.status = True

        # Inicializa Kalman com (x, y) e velocidade zero
        self.update_kalman(np.array([x, y]), timestamp)

    def updatePosition(self, x, y, r, timestamp, theta=None):
        """
        Atualiza posição da bola e aplica filtro de Kalman.
        O theta é ignorado – a direção é derivada da velocidade estimada.
        """
        self.radius = float(r)

        self.lastPosition = self.position.copy()
        self.newPosition = np.array([x, y], dtype=float)
        self.position = self.newPosition.copy()

        # --- Velocidade derivada do deslocamento (para fins de medição, mas o Kalman estima) ---
        delta = self.newPosition - self.lastPosition
        dt = max(timestamp - self.newTimestamp, 1e-3)
        if dt > 0:
            self.velocity = delta / dt  # velocidade bruta (não filtrada)

        # --- Tempo ---
        self.dT = dt
        self.oldTimestamp = self.newTimestamp
        self.newTimestamp = timestamp

        # --- Kalman usa apenas (x, y) como medição ---
        self.update_kalman(np.array([x, y]), timestamp)

        # Atualiza direção a partir da velocidade filtrada
        self._update_direction_from_velocity()

        self.updateBbox()
        self.status = True

    def setPositionNoKalman(self, x, y, r, timestamp=0.0, theta=None):
        """Atualiza posição sem tocar no Kalman (fallback)."""
        self.radius = float(r)

        self.lastPosition = self.position.copy()
        self.position = np.array([x, y], dtype=float)
        self.newPosition = self.position.copy()

        if theta is not None:
            self.theta = theta
        else:
            # mantém direção atual
            pass

        self.oldTimestamp = self.newTimestamp
        self.newTimestamp = timestamp
        self.dT = max(self.newTimestamp - self.oldTimestamp, 1e-3)

        self.updateBbox()
        self.viewBall.updateViewBot(Point2D(x, y))
        self.status = False  # não detectado

    # ======================================================================
    # 🔹 Filtro de Kalman (estado 4D: x, y, vx, vy)
    # ======================================================================

    def update_kalman(self, meas_xy, timestamp):
        """
        Atualiza o filtro com a medição de posição (x, y).
        """
        z = np.array([[meas_xy[0]], [meas_xy[1]]])

        if not self.kalman_initialized:
            self.kalman_state[0, 0] = z[0, 0]
            self.kalman_state[1, 0] = z[1, 0]
            # velocidade inicial assume zero
            self.kalman_state[2, 0] = 0.0
            self.kalman_state[3, 0] = 0.0
            self.kalman_last_time = timestamp
            self.kalman_initialized = True
            return

        dt = max(timestamp - self.kalman_last_time, 1e-3)
        self.kalman_last_time = timestamp

        # Matriz de transição (movimento de velocidade constante)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Matriz de observação (medimos apenas posição)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Predição
        self.kalman_state = F @ self.kalman_state
        self.kalman_P = F @ self.kalman_P @ F.T + self.kalman_Q

        # Inovação
        y_res = z - H @ self.kalman_state
        S = H @ self.kalman_P @ H.T + self.kalman_R
        K = self.kalman_P @ H.T @ np.linalg.inv(S)

        # Atualização
        self.kalman_state += K @ y_res
        self.kalman_P = (np.eye(4) - K @ H) @ self.kalman_P

    # ======================================================================
    # 🔹 Propriedades filtradas
    # ======================================================================

    @property
    def position_filtered(self):
        return self.kalman_state[0:2, 0]

    @property
    def velocity_filtered(self):
        return self.kalman_state[2:4, 0]

    @property
    def theta_filtered(self):
        """Orientação derivada da velocidade filtrada."""
        vx, vy = self.velocity_filtered
        return float(np.arctan2(vy, vx))

    @property
    def direction_filtered(self):
        th = self.theta_filtered
        return np.array([np.cos(th), np.sin(th)])

    # ======================================================================
    # 🔹 Previsão
    # ======================================================================

    def predict(self, timestamp):
        """
        Retorna (x_pred, y_pred, theta_pred) para compatibilidade.
        O theta_pred é derivado da velocidade predita.
        """
        if not self.kalman_initialized or self.kalman_last_time is None:
            return self.position[0], self.position[1], self.theta

        dt = timestamp - self.kalman_last_time
        if dt < 0:
            dt = 0.0
        elif dt < 1e-3:
            dt = 1e-3

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        x_pred = F @ self.kalman_state
        theta_pred = float(np.arctan2(x_pred[3, 0], x_pred[2, 0]))  # vy, vx

        return x_pred[0, 0], x_pred[1, 0], theta_pred

    # ======================================================================
    # 🔹 ROI (com covariância propagada)
    # ======================================================================

    def get_roi(self, image_shape, t_now, scale_std=3, min_size_cm=10.0):
        """
        Retorna a janela de busca em cm: (x, y, w, h) baseada na incerteza.
        """
        if not self.kalman_initialized or self.kalman_last_time is None:
            x_pred, y_pred = float(self.position[0]), float(self.position[1])
            w_cm = h_cm = float(min_size_cm)
        else:
            x_pred, y_pred, _ = self.predict(t_now)

            dt = max(t_now - self.kalman_last_time, 0.0)
            # Submatriz de posição/velocidade (índices 0,1,2,3)
            F_space = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            P_sub = self.kalman_P  # já é 4x4
            Q_sub = self.kalman_Q
            P_pred_space = F_space @ P_sub @ F_space.T + Q_sub

            std_x = float(np.sqrt(max(P_pred_space[0, 0], 0.0)))
            std_y = float(np.sqrt(max(P_pred_space[1, 1], 0.0)))

            w_cm = max(scale_std * std_x * 2.0, min_size_cm)
            h_cm = max(scale_std * std_y * 2.0, min_size_cm)

        x_cm = x_pred - w_cm / 2.0
        y_cm = y_pred - h_cm / 2.0

        return x_cm, y_cm, w_cm, h_cm

    # ======================================================================
    # 🔹 Utilitários
    # ======================================================================

    def updateBbox(self):
        self.objLimit = Circle(Point2D(self.position[0], self.position[1]), self.radius)
        self.bbox = BorderBox(GeometryType.CIRCLE, self.objLimit)

    def setImgPosition(self, xb, yb, rb):
        self.xb, self.yb, self.rb = xb, yb, rb

    def setBallColor(self, c):
        self.color = c

    # ======================================================================
    # 🔹 Reset
    # ======================================================================

    def resetState(self):
        self.direction = np.array([0.0, 0.0])
        self.status = False
        self.lastPosition = self.position.copy()
        self.newPosition = self.position.copy()
        self.updateBbox()
        self.viewBall.updateViewBot(Point2D(self.position[0], self.position[1]))

    def reset(self):
        self.position[:] = 0
        self.lastPosition[:] = 0
        self.newPosition[:] = 0
        self.direction[:] = 0
        self.velocity[:] = 0
        self.theta = 0
        self.omega = 0
        self.status = False
        self.radius = 0

        self.kalman_initialized = False
        self.kalman_state[:] = 0
        self.kalman_P = np.eye(4) * 100.0
        self.kalman_last_time = None

        self.objLimit = Circle(Point2D(0, 0), 0)
        self.updateBbox()
        self.viewBall.updateViewBot(Point2D(0, 0))