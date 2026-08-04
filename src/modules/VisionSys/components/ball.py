import numpy as np
from modules.VisionSys.components.objects import *

class Ball:
    """
    Classe da bola com suporte a direção, orientação e filtro de Kalman.
    OBS: a visão NÃO detecta theta. Ele é derivado do movimento.
    """

    def __init__(self, x=0, y=0, r=0):
        # --- Estado geométrico ---
        self.position = np.array([x, y], dtype=float)
        self.radius = float(r)

        # Direção (unitária)
        self._direction = np.array([0.0, 0.0], dtype=float)

        # Orientação (derivada *somente* do movimento)
        self._theta = 0.0  # rad

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
        self.omega = 0.0  # Velocidade angular derivada (não observada)

        # --- Kalman: estado [x, y, theta, vx, vy] (5D) ---
        # OBS: omega NÃO faz parte do vetor de estado do filtro (theta é modelado
        # como constante na transição). self.omega abaixo é apenas derivado/auxiliar.
        self.kalman_initialized = False
        self.kalman_last_time = None

        self.kalman_state = np.zeros((5, 1))
        self.kalman_P = np.eye(5) * 500.0

        self.kalman_Q = np.eye(5) * 0.05
        self.kalman_R = np.eye(3)
        self.kalman_R[0, 0] = 3.0     # x
        self.kalman_R[1, 1] = 3.0     # y
        self.kalman_R[2, 2] = 300.0   # theta é ruidoso

        # Cor da bola (médio HSV)
        self.color = None

    # ============================================================
    #         DIRECTION  <->  THETA (ANGULO)
    # ============================================================

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, d):
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

    # ======================================================================
    # 🔹 Atualização de posição
    # ======================================================================

    def setPosition(self, x, y, r, timestamp=0.0, theta=None):
        """
        Define posição inicial da bola.
        A BOLA NÃO TEM θ MEDIDO — sempre ignorar theta externo.
        """

        self.radius = float(r)

        self.position = np.array([x, y], dtype=float)
        self.lastPosition = self.position.copy()
        self.newPosition = self.position.copy()

        # direção e theta zerados, pois não há movimento
        self.direction = np.array([0.0, 0.0])
        self.theta = 0.0

        self.oldTimestamp = timestamp
        self.newTimestamp = timestamp
        self.dT = 0.0

        self.updateBbox()
        self.status = True

        # Inicializa Kalman com (x, y, θ=0)
        self.update_kalman(np.array([x, y, self.theta]), timestamp)

    def updatePosition(self, x, y, r, timestamp, theta=None):
        """
        Atualiza posição da bola e aplica filtro de Kalman.
        A bola NÃO recebe theta externo. Sempre derivamos do movimento.
        """

        self.radius = float(r)

        self.lastPosition = self.position.copy()
        self.newPosition = np.array([x, y], dtype=float)
        self.position = self.newPosition.copy()

        # --- Direção e orientação derivadas do deslocamento ---
        delta = self.newPosition - self.lastPosition
        n = np.linalg.norm(delta)

        if n > 1e-6:
            self.direction = delta / n
            self.theta = float(np.arctan2(self.direction[1], self.direction[0]))
        # se não houver deslocamento, mantém direção e theta

        # --- Tempo ---
        self.dT = max(timestamp - self.newTimestamp, 1e-3)
        self.oldTimestamp = self.newTimestamp
        self.newTimestamp = timestamp

        # --- Kalman usa x, y e theta derivado ---
        self.update_kalman(np.array([x, y, self.theta]), timestamp)

        self.updateBbox()
        self.status = True

    def setPositionNoKalman(self, x, y, r, timestamp=0.0, theta=None):
        self.radius = float(r)

        self.lastPosition = self.position.copy()
        self.position = np.array([x, y], dtype=float)
        self.newPosition = self.position.copy()

        if theta is not None:
            self.theta = theta   # atualiza direction internamente
        else:
            # mantém direção atual
            pass

        self.oldTimestamp = self.newTimestamp
        self.newTimestamp = timestamp
        self.dT = max(self.newTimestamp - self.oldTimestamp, 1e-3)

        self.updateBbox()
        self.viewBall.updateViewBot(Point2D(x, y))

        # NÃO CHAMA update_kalman
        self.status = False

    # ======================================================================
    # 🔹 Filtro de Kalman (estado completo)
    # ======================================================================

    def update_kalman(self, meas_xyz, timestamp):
        mx, my, mtheta = meas_xyz.reshape(3,)
        z = np.array([[mx], [my], [mtheta]])

        if not self.kalman_initialized:
            self.kalman_state[0, 0] = mx
            self.kalman_state[1, 0] = my
            self.kalman_state[2, 0] = mtheta
            self.kalman_last_time = timestamp
            self.kalman_initialized = True
            return

        dt = max(timestamp - self.kalman_last_time, 1e-3)
        self.kalman_last_time = timestamp

        F = np.array([
            [1, 0, 0, dt, 0],
            [0, 1, 0, 0, dt],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])

        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])

        # Predição
        self.kalman_state = F @ self.kalman_state
        self.kalman_P = F @ self.kalman_P @ F.T + self.kalman_Q

        # Inovação
        y_res = z - (H @ self.kalman_state)
        y_res[2, 0] = (y_res[2, 0] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.kalman_P @ H.T + self.kalman_R
        K = self.kalman_P @ H.T @ np.linalg.inv(S)

        # Atualização
        self.kalman_state += K @ y_res
        
        # [CORREÇÃO] Normaliza o ângulo interno do vetor de estado para evitar estouro de escala
        self.kalman_state[2, 0] = (self.kalman_state[2, 0] + np.pi) % (2 * np.pi) - np.pi
        self.theta = float(self.kalman_state[2, 0])


    # ======================================================================
    # 🔹 Propriedades filtradas
    # ======================================================================

    @property
    def position_filtered(self):
        return self.kalman_state[0:2, 0]

    @property
    def theta_filtered(self):
        return self.kalman_state[2, 0]

    @property
    def direction_filtered(self):
        th = self.theta_filtered
        return np.array([np.cos(th), np.sin(th)])

    @property
    def velocity_filtered(self):
        return self.kalman_state[3:5, 0]

    # ======================================================================
    # 🔹 Previsão
    # ======================================================================
    def predict(self, timestamp):
        # Se o Kalman nunca foi inicializado, devolve estado atual sem previsão
        if not self.kalman_initialized or self.kalman_last_time is None:
            return self.position[0], self.position[1], self.theta

        # dt relativo ao último UPDATE real, não altera estado
        dt = timestamp - self.kalman_last_time
        if dt < 0:
            dt = 0.0
        elif dt < 1e-3:
            dt = 1e-3

        F = np.array([
            [1, 0, 0, dt, 0],
            [0, 1, 0, 0, dt],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])

        # Predição *sem alterar o filtro*
        x_pred = F @ self.kalman_state

        return x_pred[0,0], x_pred[1,0], x_pred[2,0]




    def get_roi(self, image_shape, t_now, vision_sys, scale_std=3):
        """
        Retorna as dimensões do ROI centrado na previsão do Kalman, convertendo cm para pixels.
        """
        if not self.kalman_initialized or self.kalman_last_time is None:
            return 0, 0, image_shape[1], image_shape[0] # Retorna imagem inteira se não inicializado

        # --- 1) Posição predita pelo Kalman (Em Centímetros) ---
        x_pred, y_pred, _ = self.predict(t_now)

        # --- 2) CONVERSÃO CRÍTICA: Centímetros globais -> Pixels Absolutos da Imagem ---
        x_img, y_img = vision_sys.getImageRealIndice([x_pred, y_pred])

        # --- 3) AJUSTE DE RECORTE: Tornar o pixel relativo à subimagem fieldReduce ---
        if vision_sys.viewCapture.cooVetor is not None:
            x_offset, y_offset = vision_sys.viewCapture.cooVetor[0], vision_sys.viewCapture.cooVetor[1]
            x_c = x_img - x_offset
            y_c = y_img - y_offset
        else:
            x_c, y_c = x_img, y_img

        # --- 4) FATOR DE ESCALA DINÂMICO: Pixels por Centímetro ---
        # Baseado no último raio da bola mapeado em pixels (self.rb) vs raio real (self.radius)
        pixels_per_cm = (self.rb / self.radius) if (hasattr(self, 'rb') and self.rb > 0) else 5.0

        # --- 5) PROPAÇÃO DA COVARIÂNCIA (Incerteza acumulada no tempo dt) ---
        dt = max(t_now - self.kalman_last_time, 0.0)
        F_space = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Isola submatrizes de posição/velocidade [x, y, vx, vy] da bola para projetar P_pred
        P_sub = self.kalman_P[[0,1,3,4], :][:, [0,1,3,4]]
        Q_sub = self.kalman_Q[[0,1,3,4], :][:, [0,1,3,4]]
        P_pred_space = F_space @ P_sub @ F_space.T + Q_sub

        # Desvio padrão convertido de centímetros para PIXELS
        std_x = np.sqrt(P_pred_space[0, 0]) * pixels_per_cm
        std_y = np.sqrt(P_pred_space[1, 1]) * pixels_per_cm

        # --- 6) Define tamanho do ROI em pixels ---
        w_roi = int(scale_std * std_x * 2)
        h_roi = int(scale_std * std_y * 2)

        # [CORREÇÃO] Margem de segurança baseada no tamanho real da bola na tela
        min_dim = int(3.5 * self.rb) if (hasattr(self, 'rb') and self.rb > 0) else 35
        w_roi = max(w_roi, min_dim)
        h_roi = max(h_roi, min_dim)
        
        # --- 7) Topo-esquerdo do ROI ---
        x = int(x_c - w_roi // 2)
        y = int(y_c - h_roi // 2)

        # --- 8) Ajusta limites às dimensões da imagem enviada ---
        h_img, w_img = image_shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w_roi = min(w_roi, w_img - x)
        h_roi = min(h_roi, h_img - y)

        return x, y, w_roi, h_roi

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
        self.kalman_P = np.eye(5) * 500
        self.kalman_last_time = None

        self.objLimit = Circle(Point2D(0, 0), 0)
        self.updateBbox()
        self.viewBall.updateViewBot(Point2D(0, 0))
