# Updated Robot class with simple Kalman for [x, y, theta]
# (Canvas version; user requested refactor while keeping old API)

import numpy as np
import cv2
from modules.VisionSys.components.objects import *

class Robot:
    def __init__(self, id: ID_Robots, team: ID_Team, x=0, y=0, r=0,
                 image=cv2.imread('src/data/images/dark_screen.png'),
                 colorTeam=None, colorCar1=None, colorCar2=None,
                 differential_filter=False):
        # Identifiers
        self.id = id
        self.team = team

        # Position
        self.position = np.array([float(x), float(y)])
        self.lastPosition = self.position.copy()
        self.newPosition = self.position.copy()

        # Direction & angle
        self._direction = np.array([1.0, 0.0])
        self._theta = 0.0

    
        # Geometry
        self.radius = float(r)
        self.objLimit = Circle(Point2D(x, y), self.radius)
        self.bbox = BorderBox(GeometryType.CIRCLE, self.objLimit)
        self.ObjType = ObjTypeMove.MOVING
        self.objTypeSystem = ObjTypeVision.ROBOT

        # Variáveis internas do robô
        self.axle_length = 2*self.radius #cm

        # Status
        self.detected = False
        self.possessionBall = False

        # Image-space coordinates (DO NOT CHANGE)
        self.xi = 0
        self.yi = 0
        self.ri = 0

        # Image & view window
        self.image = image
        self.dimMatrix = image.shape[1] if image is not None else 0
        self.viewRect = ViewBot(Point2D(self.position[0], self.position[1]), self.dimMatrix)

        # Colors
        self.colorTeam = colorTeam
        self.colorCar1 = colorCar1
        self.colorCar2 = colorCar2

        # Time
        self.lastTimestamp = 0
        self.newTimestamp = 0
        self.dT = 0.0

        # Kalman
        self.kalman_initialized = False
        self.kalman_last_time = None
        self._init_kalman()

    # --------------------------
    # Kalman init
    # --------------------------
    def _init_kalman(self):
        # ESTADO NOVO: [x, y, theta, v_esquerda (vL), v_direita (vR), omega]
        self.kalman_state = np.zeros((6, 1), float)

        # Covariances
        self.kalman_P = np.eye(6) * 400.0

        # Processo (Q): [pos, pos, ang, vL_noise, vR_noise, omega_noise]
        # Processo (acelerações pequenas)
        self.kalman_Q = np.diag([0.01, 0.01, 0.01, 5.0, 5.0, 1.0])

        # Ruído de medição do sistema de visão
        # Medimos apenas x, y, theta  → tamanho 3x3
        self.kalman_R = np.diag([3.0, 3.0, 0.5])

        self.kalman_initialized = False
        self.kalman_last_time = None

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
        self._theta = np.arctan2(self._direction[1], self._direction[0])

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, ang):
        self._theta = float(ang)
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
        if not self.kalman_initialized:
            return np.array([0.0, 0.0])
        return self.kalman_state[3:5, 0]

    @property
    def omega_filtered(self):
        if not self.kalman_initialized:
            return 0.0
        return float(self.kalman_state[5, 0])


    # --------------------------
    # Position update
    # --------------------------
    def setPosition(self, x, y, direction, image, time, wheel_velocities=None):
        self.lastTimestamp = self.newTimestamp
        self.newTimestamp = time
        self.dT = self.newTimestamp - self.lastTimestamp if self.lastTimestamp != 0 else 0.0

        # Position update (with noise guard)
        self.lastPosition = self.position.copy()
        raw_move = np.array([x - self.lastPosition[0], y - self.lastPosition[1]])
        if np.linalg.norm(raw_move) > 0.1:  # 0.1 cm threshold
            self.position = np.array([x, y], float)
        # else: ignore noise

        self.newPosition = self.position.copy()

        self.direction = direction 
        self.image = image
        if image is not None:
            self.viewRect.setDimension(image.shape[1])

        # Geometry
        self.objLimit = Circle(Point2D(x, y), self.radius)
        self.viewRect.updateViewBot(Point2D(x, y))
        self.updateBbox()

        # Update Kalman
        self.update_kalman([self.position[0], self.position[1], self.theta], time)


    def updatePosition(self, x, y, direction, image, time, wheel_velocities=None):
        self.setPosition(x, y, direction, image, time)

    def setRadius(self, r):
        self.radius = r

    def setPositionNoKalman(self, x, y, theta, timestamp, image=None):
        '''
            Atualizo a posição do robô sem realziar a predição
        '''
        self.lastPosition = self.position.copy()
        self.position = np.array([x, y], float)
        self.newPosition = self.position.copy()

        self.theta = theta     # atualiza direction automaticamente

        self.lastTimestamp = self.newTimestamp
        self.newTimestamp = timestamp
        self.dT = max(self.newTimestamp - self.lastTimestamp, 1e-3)

        if image is not None:
            self.image = image
            self.viewRect.setDimension(image.shape[1])

        self.objLimit = Circle(Point2D(x, y), self.radius)
        self.updateBbox()
        self.viewRect.updateViewBot(Point2D(x, y))

        # NADA de update do kalman
        self.detected = False

    # --------------------------
    # Kalman update
    # --------------------------
    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        return (d + np.pi) % (2*np.pi) - np.pi

    # ------------------------
    # EKF: Funções de predição (Novas)
    # ------------------------
    def _non_linear_motion_model(self, state, dt):
        """
        Função de transição não-linear f(x, dt) para o modelo Differential Drive.
        Assume que w_esquerda = w_direita = 0 (sem entrada de comando).
        """
        x, y, theta, vL, vR, omega = state[:, 0]
        
        # Velocidade Linear (v) e Angular (omega)
        v = (vL + vR) / 2.0
        # O estado omega (x[5]) já representa a velocidade angular filtrada.
        # omega = (vR - vL) / self.axle_length # Se usássemos vL e vR para a predição de omega

        # Aproximação de Euler (simplificada para pequenos dt)
        x_new = x + v * dt * np.cos(theta)
        y_new = y + v * dt * np.sin(theta)
        theta_new = theta + omega * dt # Predição de theta usando o estado omega
        
        # Velocidades são assumidas como Constantes (CV)
        vL_new = vL
        vR_new = vR
        omega_new = omega

        # Normaliza theta
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([[x_new], [y_new], [theta_new], [vL_new], [vR_new], [omega_new]])

    def _jacobian_motion_model(self, state, dt):
        """
        Matriz Jacobiana A (ou F) do modelo de movimento não-linear f(x, dt).
        """
        x, y, theta, vL, vR, omega = state[:, 0]
        v = (vL + vR) / 2.0
        
        # Derivadas parciais para x, y, theta
        dx_dtheta = -v * dt * np.sin(theta)
        dy_dtheta = v * dt * np.cos(theta)
        
        dx_dvL = 0.5 * dt * np.cos(theta) # d(x_new)/d(vL) = d(v)/d(vL) * dt * cos(theta)
        dy_dvL = 0.5 * dt * np.sin(theta) # d(y_new)/d(vL)
        
        dx_dvR = 0.5 * dt * np.cos(theta) # d(x_new)/d(vR)
        dy_dvR = 0.5 * dt * np.sin(theta) # d(y_new)/d(vR)
        
        # Matriz Jacobiana 6x6 (A)
        A = np.array([
            # x   y   theta       vL          vR          omega
            [1, 0, dx_dtheta,   dx_dvL,     dx_dvR,     0],
            [0, 1, dy_dtheta,   dy_dvL,     dy_dvR,     0],
            [0, 0, 1,           0,          0,          dt],
            [0, 0, 0,           1,          0,          0],
            [0, 0, 0,           0,          1,          0],
            [0, 0, 0,           0,          0,          1]
        ], float)
        
        return A

    def update_kalman(self, z_list, timestamp):
        x, y, theta_meas = z_list
        z = np.array([[x],[y],[theta_meas]])

        if not self.kalman_initialized:
            self.kalman_state[:3,0] = [x, y, theta_meas]
            self.kalman_initialized = True
            self.kalman_last_time = timestamp
            return

        dt = max(timestamp - self.kalman_last_time, 1e-3)
        self.kalman_last_time = timestamp

        # ---- Predição EKF (Fase 1) ---

        # 1. Predição do Estado (função não-linear F)
        self.kalman_state = self._non_linear_motion_model(self.kalman_state, dt)

        # 2. Predição da Covariância (usando o Jacobiano A)
        A = self._jacobian_motion_model(self.kalman_state, dt)
        self.kalman_P = A @ self.kalman_P @A.T + self.kalman_Q

        # --- CORREÇÃO EKF (Fase 2) ---
        H = np.array([
            [1, 0, 0, 0, 0, 0],   # mede x
            [0, 1, 0, 0, 0, 0],   # mede y
            [0, 0, 1, 0, 0, 0],   # mede theta
        ], float)

        # H é linear, por isso usamos H @ x_predito
        pred = H @ self.kalman_state 
        y_residual = z-pred 
        # normalize theta residual
        y_residual[2,0] = self._angle_diff(z[2,0], pred[2,0])

        S = H @ self.kalman_P @ H.T + self.kalman_R
        K = self.kalman_P @ H.T @ np.linalg.inv(S)

        self.kalman_state = self.kalman_state + K @ y_residual
        self.kalman_P = (np.eye(6) - K @ H) @ self.kalman_P

    # Preciso realizar a conversão desses valores para a coordenada da imagem
    def get_roi(self, image_shape, t_now, vision_sys, scale_std=3):
        if not self.kalman_initialized or self.kalman_last_time is None:
            return 0, 0, image_shape[1], image_shape[0]

        # --- 1) Posição predita pelo EKF (Em Centímetros) ---
        st_pred, P_pred = self.predict_with_cov(t_now)
        x_pred = st_pred[0, 0]
        y_pred = st_pred[1, 0]

        # --- 2) CONVERSÃO CRÍTICA: Centímetros globais -> Pixels Absolutos ---
        x_img, y_img = vision_sys.getImageRealIndice([x_pred, y_pred])
        
        # --- 3) AJUSTE DE RECORTE: Relativo à subimagem fieldReduce ---
        if vision_sys.viewCapture.cooVetor is not None:
            x_offset, y_offset = vision_sys.viewCapture.cooVetor[0], vision_sys.viewCapture.cooVetor[1]
            x_c = x_img - x_offset
            y_c = y_img - y_offset
        else:
            x_c, y_c = x_img, y_img

        # --- 4) FATOR DE ESCALA DINÂMICO: Pixels por Centímetro ---
        # Baseado no raio atual do robô em pixels (self.ri) vs raio real em cm (self.radius)
        pixels_per_cm = (self.ri / self.radius) if self.ri > 0 else 5.0

        # --- 5) Desvios padrão convertidos de centímetros para PIXELS ---
        std_x = np.sqrt(P_pred[0, 0]) * pixels_per_cm
        std_y = np.sqrt(P_pred[1, 1]) * pixels_per_cm

        # --- 6) Define tamanho do ROI em pixels ---
        w_roi = int(scale_std * std_x * 2)
        h_roi = int(scale_std * std_y * 2)

        # [CORREÇÃO] Garante tamanho mínimo coerente com o tamanho do robô em pixels na tela
        min_dimension = int(3.0 * self.ri) if self.ri > 0 else int(2.5 * self.radius * pixels_per_cm)
        w_roi = max(w_roi, min_dimension)
        h_roi = max(h_roi, min_dimension)

        # --- 7) Topo-esquerdo ---
        x = int(x_c - w_roi // 2)
        y = int(y_c - h_roi // 2)

        # --- 8) Ajusta limites à imagem ---
        h_img, w_img = image_shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w_roi = min(w_roi, w_img - x)
        h_roi = min(h_roi, h_img - y)

        return x, y, w_roi, h_roi
    
    # Recuperar predição do filtro de Kalman
    def predict(self, time):
        """
        Prediz o estado futuro usando o timestamp absoluto (AGORA USANDO O MODELO EKF NÃO-LINEAR).
        Não altera o estado interno do filtro, apenas retorna a predição.
        """
        if not self.kalman_initialized:
            # Retorna o estado cru se o filtro não foi inicializado
            return self.position[0], self.position[1], self.theta

        # tempo entre a última atualização real e a predição desejada
        dt = max(time - self.kalman_last_time, 0.0)

        # UTILIZA O MODELO DE MOVIMENTO NÃO-LINEAR (EKF)
        # st_predicted é um vetor (6, 1) com [x, y, theta, vL, vR, omega]
        st_predicted = self._non_linear_motion_model(self.kalman_state, dt)

        # Retorna apenas as coordenadas de posição (x, y, theta)
        return st_predicted[0,0], st_predicted[1,0], st_predicted[2,0]

    def predict_with_cov(self, time):
        """
        Retorna (state_pred, P_pred) sem alterar estado interno.
        AGORA USA O MODELO NÃO-LINEAR E O JACOBIANO.
        """
        if not self.kalman_initialized:
            st = np.zeros((6,1))
            st[:3,0] = [self.position[0], self.position[1], self.theta]
            P_temp = np.eye(6) * 50.0  
            return st, P_temp


        dt = max(time - self.kalman_last_time, 0.0)
        
        # Predição de estado
        st_pred = self._non_linear_motion_model(self.kalman_state, dt)
        
        # Predição de covariância
        A = self._jacobian_motion_model(st_pred, dt) # Usa o estado predito para o jacobiano (comum em EKF)
        P_pred = A @ self.kalman_P @ A.T + self.kalman_Q

        # normalize theta (já feito em _non_linear_motion_model, mas seguro repetir)
        st_pred[2,0] = (st_pred[2,0] + np.pi) % (2*np.pi) - np.pi
        return st_pred, P_pred
    
    # --------------------------
    # Aux
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
        """
        Define a cor lógica do time do robô.
        Mantém API antiga, não interfere na física ou posição.
        """
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

