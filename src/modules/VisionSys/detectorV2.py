# ==========================================================================================
# MÓDULO DE DETECÇÃO VSS (Vision System Soccer) -- v3.3.22 (BETA)
# ==========================================================================================
import cv2
import numpy as np
from timer import *
import os

from modules.VisionSys.components.objects import *
from modules.VisionSys.components.viewcapture import *
from modules.VisionSys.components.field import *
from modules.VisionSys.components.ball import *
from modules.VisionSys.components.robot import *
from modules.VisionSys.components.combination import *

from modules.control.comm.protocols import common_pb2

import traceback

# ====================== SISTEMA DE VISÃO (VSS) ==================
class VisionSystem:
    '''Processa a imagem da câmera e extrai bola/robôs/campo para o resto do sistema.'''

    def __init__(self, config: EConfig = EConfig(), debug: bool = False,
                capture: Capture = None, UseCuda: bool = False, GPUType: GPUType = None):
        """
        Inicializa o sistema de visão com todos os parâmetros, kernels e cache.
        """
        # ==============================================================
        # 1. OBJETOS BASE E CONTADORES
        # ==============================================================
        self.CreateObjs()
        self._countProcess = 0
        self.bmk = Benchmark()

        # Watchdog: frames perdidos antes de resetar o Kalman
        self.MAX_MISSED_FRAMES_BALL = 180   # 2s a 60 FPS, 4s a 30 FPS
        self.MAX_MISSED_FRAMES_ROBOT = 180
        self.missed_frames_ball = 0
        self.ball_kalman_reset_flag = False
        self.missed_frames_robots = {}
        self.robot_kalman_reset_flags = {}


        ## Variável de debug temporária
        self.var_d = False 

        # Gating de distância – rejeita saltos implausíveis
        self.MAX_JUMP_CM = 30.0

        self._count: int = 0
        self._firstTimeExec: int = 0
        self.config: EConfig = config
        self._capture: Capture = capture
        self._hasCuda = UseCuda
        self._GPUType = GPUType

        if self._capture is not None:
            self._capture.GPUMode(self._hasCuda)

        # ==============================================================
        # 2. ESTADO DA CÂMERA / CUDA
        # ==============================================================
        self.GPUimg = None
        self.CPUimg = None
        self.emulatorMode = MODE_IMAGE          # imagem estática por padrão

        self.timer: HighPrecisionTimer = None   # timer de alta precisão
        self.dT = 0                             # intervalo de tempo entre frames

        if self._hasCuda:
            self.GPUimg = cv2.cuda.GpuMat()

        # ==============================================================
        # 3. GEOMETRIA DO CAMPO E HOMOGRAFIA
        # ==============================================================
        self.fieldWidth = 150                   # largura do campo (cm)
        self.fieldHeight = 130                  # altura do campo (cm)
        self.prop_px_cm = 1                     # proporção pixel → cm (atualizada na detecção)
        self.prop_px_cm_virtual = 3             # proporção na imagem virtual (fixa)
        self.min_diag = (7.5 / 4) * np.sqrt(2) * self.prop_px_cm

        self.homography_matrix = None           # matriz de homografia (real → virtual)
        self.inv_homography_matrix = None       # inversa (virtual → real)

        self.debug: bool = debug

        # Coordenada da origem do sistema de coordenadas O' (pixels na virtual)
        self.xnv = 67
        self.ynv = 402
        self.coordOrigin = np.array([self.xnv, self.ynv])

        # Controle de tempo de execução
        self.lastMajorTime = 0
        self.currentTime = 0
        self.newSendTime = 0.02

        # Tamanho padrão da bola (cm)
        self.ballRadiusP = 2.135

        self.modDpCm = 0
        self.fieldDetectedFlag = False

        # Pontos fixos da imagem virtual (em pixels)
        # Extremos do campo
        self.fieldP1v = np.array([97, 12])
        self.fieldP2v = np.array([547, 12])
        self.fieldP3v = np.array([547, 402])
        self.fieldP4v = np.array([97, 402])
        self.fieldCenterv = np.array([322, 207])

        # Pivots (áreas dos times)
        self.PA1v = np.array([210, 87])
        self.PA2v = np.array([210, 207])
        self.PA3v = np.array([210, 327])
        self.PE1v = np.array([435, 87])
        self.PE2v = np.array([435, 207])
        self.PE3v = np.array([435, 327])

        # Áreas do goleiro aliado (externa e interna)
        self.GA1v  = np.array([97, 102])
        self.GA2v  = np.array([142, 102])
        self.GA3v  = np.array([142, 312])
        self.GA4v  = np.array([97, 312])
        self.GAI1v = np.array([67, 147])
        self.GAI2v = np.array([97, 147])
        self.GAI3v = np.array([97, 267])
        self.GAI4v = np.array([67, 267])

        # Áreas do goleiro inimigo (externa e interna)
        self.GE1v  = np.array([502, 102])
        self.GE2v  = np.array([547, 102])
        self.GE3v  = np.array([547, 312])
        self.GE4v  = np.array([502, 312])
        self.GEI1v = np.array([547, 147])
        self.GEI2v = np.array([577, 147])
        self.GEI3v = np.array([577, 267])
        self.GEI4v = np.array([547, 267])

        # Pontos médios dos lados do campo
        self.fieldP12v = np.array([322, 12])
        self.fieldP34v = np.array([322, 402])

        # ==============================================================
        # 4. PARÂMETROS DE PROCESSAMENTO DE IMAGEM
        # ==============================================================
        self.offSetWindow = 10
        self.offSetErode = 0
        self.dimMatrix = 25
        self.Thrashhold = 235
        self.pixelWidth = 1


        ##=========================================================
        # Helpers de pixel 
        self.winSize = 0 
        self.half_win = 0
        self.playerRadius = 0
        self.mainColorRadius = 0
        self.secColorRadius = 0
        self.single_area = 0
        self.NOISE_RADIUS_MIN = 0 

        ##=========================================================
        # ==============================================================
        # 5. IMAGENS E BUFFERS DE DEBUG
        # ==============================================================
        self.frameOrigin = None
        self.ballImg = None
        self.fieldReduce = None
        self.frameResult = None
        self.imgReduce = None
        self.binaryAllies = None
        self.binaryAllyConfirmed = None

        self.virtual = cv2.imread("src/data/images/CampoVirtual.png")
        self.virtualImg = self.virtual.copy()

        # Imagens binarizadas (serão redimensionadas depois)
        self.binaryObjects = np.zeros((1, 1), dtype=np.uint8)
        self.binaryPlayers = np.zeros((1, 1), dtype=np.uint8)
        self.binaryBall = np.zeros((1, 1), dtype=np.uint8)
        self.binReduceField = np.zeros((1, 1), dtype=np.uint8)
        self.binField = np.zeros((1, 1), dtype=np.uint8)

        # ==============================================================
        # 6. CORES E LIMITES
        # ==============================================================
        self.ballColor = None
        self.allyColor = None
        self.enemyColor = None
        self.goalAllyColor1 = None
        self.goalAllyColor2 = None
        self.atk1AllyColor1 = None
        self.atk1AllyColor2 = None
        self.atk2AllyColor1 = None
        self.atk2AllyColor2 = None

        # Limites globais para objetos
        self.objectsDarkColor = np.array([0, 10, 130])
        self.objectsLightColor = np.array([179, 255, 255])

        # Raios (serão definidos em tempo de execução)


        # Limites HSV para aliados e inimigos (preenchidos no ToMineData)
        self.ally_lower_bound = None
        self.ally_upper_bound = None
        self.enemy_lower_bound = None
        self.enemy_upper_bound = None

        # ==============================================================
        # 7. ESTRUTURAS DE DADOS E CONTROLE
        # ==============================================================
        self.playersCount = 0
        self.alliesCount = 0
        self.enemiesCount = 0
        self.fieldDetectionFailCount = 0
        self.maxFieldFailures = 10

        self.playersWindows = [None, None, None, None, None, None]
        self.alliesWindows = [None, None, None]
        self.enimiesWindows = [None, None, None]

        self._threads = []
        self.newProcTime = 10000

        self.colorTree = TreeColors()

        # Carrega as configurações do emulador
        self.ToMineData()

        # ==============================================================
        # 8. CACHE DE KERNELS E ESTRUTURAS MORFOLÓGICAS
        # ==============================================================
        # Kernels para detecção de campo (blur, morfologia)
        self.kernel_blur = (5, 5)
        self.kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Buffers para homografia
        self.ptsSource_buffer = np.zeros((4, 2), dtype=np.float32)
        self.ptsSource_cache = np.zeros((4, 2), dtype=np.float32)
        self.ptsFinal_cache = np.array([
            self.fieldP1v, self.fieldP2v,
            self.fieldP3v, self.fieldP4v
        ], dtype=np.float32)

        # Kernels morfológicos padrão (reutilizados em todo o código)
        self.struct_ellipse3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.struct_ellipse5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.struct_ellipse7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.struct_ellipse11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.struct_ellipse13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

        self.struct_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.struct_rect5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.struct_rect7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        self.struct_rect11 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        self.struct_rect13 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        
        # ==============================================================
        # 9. CONSTRUÇÃO DO CAMPO (após todos os parâmetros)
        # ==============================================================
        self.BuildField()
        
    # ==========================================================================================
    # BLOCO 1: INICIALIZAÇÃO, CONFIGURAÇÃO E RESET
    # ==========================================================================================
    def CreateObjs(self):
        '''Cria os objetos do sistema (bola, robôs, campo) ainda sem dados cadastrados.'''
        self.ball = Ball()

        self.robotAllyG = Robot(id=ID_Robots.ROBOT_ALLY_GOAL, team=ID_Team.TEAM_ALLY)
        self.robotAlly1 = Robot(id=ID_Robots.ROBOT_ALLY_1,team=ID_Team.TEAM_ALLY)
        self.robotAlly2 = Robot(id=ID_Robots.ROBOT_ALLY_2,team=ID_Team.TEAM_ALLY)
        
        #robôs inimigos
        self.robotEnemyG = Robot(id=ID_Robots.ROBOT_ENEMY_GOAL,team=ID_Team.TEAM_ENEMY)
        self.robotEnemy1 = Robot(id=ID_Robots.ROBOT_ENEMY_1,team=ID_Team.TEAM_ENEMY)
        self.robotEnemy2 = Robot(id=ID_Robots.ROBOT_ENEMY_2,team=ID_Team.TEAM_ENEMY)

        #Dividindo times
        #time aliado
        self.allyTeam = [self.robotAllyG, self.robotAlly1, self.robotAlly2]

        #time inimigo
        self.enemyTeam = [self.robotEnemyG,self.robotEnemy1,self.robotEnemy2]

        #gerando objeto para representar o campo
        self.field = Field(self)

        #gerando a viewCapture
        self.viewCapture = ViewCapture()

    def BuildField(self):
        '''Converte os pontos fixos do campo virtual (pixels) para cm no sistema O' e monta o objeto Field.'''
        virtualExtrems = Quad(P1 = self.GetPointVirtual(Point2D(self.fieldP1v[0],self.fieldP1v[1])),
                              P2 = self.GetPointVirtual(Point2D(self.fieldP2v[0],self.fieldP2v[1])),
                              P3 = self.GetPointVirtual(Point2D(self.fieldP3v[0],self.fieldP3v[1])),
                              P4 = self.GetPointVirtual(Point2D(self.fieldP4v[0],self.fieldP4v[1])))

        self.field.virtualExtrems(virtualExtrems)

        #adicionando pivots (Convertendo todos para cm)
        x, y = self.GetPointVirtual(self.PA1v)
        self.field.setPivotPos(ID_Pivots.PA1, x,y)

        x, y = self.GetPointVirtual(self.PA2v)
        self.field.setPivotPos(ID_Pivots.PA2, x,y)

        x, y = self.GetPointVirtual(self.PA3v)
        self.field.setPivotPos(ID_Pivots.PA3, x,y)

        x, y = self.GetPointVirtual(self.PE1v)
        self.field.setPivotPos(ID_Pivots.PE1, x,y)

        x, y = self.GetPointVirtual(self.PE2v)
        self.field.setPivotPos(ID_Pivots.PE2, x,y)

        x, y = self.GetPointVirtual(self.PE3v)
        self.field.setPivotPos(ID_Pivots.PE3, x,y)

        x, y = self.GetPointVirtual(self.fieldCenterv)
        self.field.setPivotPos(ID_Pivots.CENTER, x,y)
        
        
        #gerando áreas do gol onde os goleiros ficarão
        #areas do goleiro aliado
        goalAllyArea = Quad(P1 = self.GetPointVirtual(Point2D(self.GA1v[0],self.GA1v[1])),
                              P2 = self.GetPointVirtual(Point2D(self.GA2v[0],self.GA2v[1])),
                              P3 = self.GetPointVirtual(Point2D(self.GA3v[0],self.GA3v[1])),
                              P4 = self.GetPointVirtual(Point2D(self.GA4v[0],self.GA4v[1])))
        
        self.field.setAreaRobotGoal(id=ID_Field.GOAL_AREA_ALLY,rect= goalAllyArea)


        #áreas do goleiro inimigo
        goalEnemyArea = Quad(P1 = self.GetPointVirtual(Point2D(self.GE1v[0],self.GE1v[1])),
                              P2 = self.GetPointVirtual(Point2D(self.GE2v[0],self.GE2v[1])),
                              P3 = self.GetPointVirtual(Point2D(self.GE3v[0],self.GE3v[1])),
                              P4 = self.GetPointVirtual(Point2D(self.GE4v[0],self.GE4v[1])))
        
        self.field.setAreaRobotGoal(id=ID_Field.GOAL_AREA_ENEMY, rect=goalEnemyArea)

        #gerando áreas internas dos gols onde serão pontuados
        #areas do goleiro aliado
        goalAlly = Quad(P1 = self.GetPointVirtual(Point2D(self.GAI1v[0],self.GAI1v[1])),
                        P2 = self.GetPointVirtual(Point2D(self.GAI2v[0],self.GAI2v[1])),
                        P3 = self.GetPointVirtual(Point2D(self.GAI3v[0],self.GAI3v[1])),
                        P4 = self.GetPointVirtual(Point2D(self.GAI4v[0],self.GAI4v[1])))
        
        self.field.setAreaGoal(id=ID_Field.GOAL_ALLY, rect=goalAlly)


        #áreas do goleiro inimigo
        goalEnemy = Quad(P1 = self.GetPointVirtual(Point2D(self.GEI1v[0],self.GEI1v[1])),
                         P2 = self.GetPointVirtual(Point2D(self.GEI2v[0],self.GEI2v[1])),
                         P3 = self.GetPointVirtual(Point2D(self.GEI3v[0],self.GEI3v[1])),
                         P4 = self.GetPointVirtual(Point2D(self.GEI4v[0],self.GEI4v[1])))
        
        self.field.setAreaGoal(id=ID_Field.GOAL_ENEMY, rect=goalEnemy)

    def ToMineData(self):
        '''Extrai as configurações do objeto EConfig (emulador) para o sistema de visão.'''
        print("[VisSys][ToMineData][INFO] - Configurações carregadas")
        self.offSetWindow   = self.config.offSetWindow  
        self.offSetErode    = self.config.offSetErode   
        self.dimMatrix      = self.config.dimMatrix     
        self.Trashhold      = self.config.Trashhold     
        self.fieldWidth     = self.config.fieldWidth    
        self.fieldHeight    = self.config.fieldHeight   
        self.ballColor      = self.config.ballColor     
        self.allyColor      = self.config.allyColor     
        self.enemyColor     = self.config.enemyColor     
        self.goalAllyColor1 = self.config.goalAllyColor1
        self.goalAllyColor2 = self.config.goalAllyColor2
        self.atk1AllyColor1 = self.config.atk1AllyColor1
        self.atk1AllyColor2 = self.config.atk1AllyColor2 
        self.atk2AllyColor1 = self.config.atk2AllyColor1
        self.atk2AllyColor2 = self.config.atk2AllyColor2
        self.emulatorMode   = self.config.emulatorMode
        self.timer          = self.config.timer

        #atribuindo cores principais aos robôs
        self.robotAlly1.setTeamColor(self.allyColor)
        self.robotAlly2.setTeamColor(self.allyColor)
        self.robotAllyG.setTeamColor(self.allyColor)

        self.robotEnemy1.setTeamColor(self.enemyColor)
        self.robotEnemy2.setTeamColor(self.enemyColor)
        self.robotEnemyG.setTeamColor(self.enemyColor)

        # Limites de cor
        self.ally_lower_bound, self.ally_upper_bound = self.CreateColorBounds(self.allyColor)
        self.enemy_lower_bound, self.enemy_upper_bound = self.CreateColorBounds(self.enemyColor)

        #Cor laranja da bola (mesmo tratamento de wrap de Hue, com tolerância menor)
        self.ball_lower_bound, self.ball_upper_bound = self.CreateColorBounds(
            self.ballColor, hue_tolerance=6, saturation_tolerance=50, value_tolerance=50)

        #puxando endereços de comunicação protobuff
        self.iPPbReceive  = self.config.ip_send
        self.portPbReceive = self.config.port_send
        self.iPPbSend     = self.config.ip_receive
        self.portPbSend   = self.config.port_receive

        self.SetTreeColorDefault()

    def SetConfigEmulator(self, config:EConfig):
        '''Aplica uma nova configuração vinda do emulador.'''
        self.config = config
        self.ToMineData()

    def SetTreeColorDefault(self):
        '''Popula a ColorTree com as cores fixas dos aliados (time + primária + secundária).'''
        self.colorTree.add_robot(
            ID_Team.TEAM_ALLY,
            ID_Robots.ROBOT_ALLY_GOAL,
            self.allyColor,
            self.goalAllyColor1,
            self.goalAllyColor2
        )
        
        self.colorTree.add_robot(
            ID_Team.TEAM_ALLY,
            ID_Robots.ROBOT_ALLY_1,
            self.allyColor, 
            self.atk1AllyColor1,
            self.atk1AllyColor2
        )
        
        self.colorTree.add_robot(
            ID_Team.TEAM_ALLY,
            ID_Robots.ROBOT_ALLY_2, 
            self.allyColor, 
            self.atk2AllyColor1,
            self.atk2AllyColor2
        )
        
        # Time Inimigo (exemplo - se necessário)
        # self.colorTree.add_robot(
        #     ID_Team.TEAM_ENEMY,
        #     self.enemyColor,
        #     ID_Robots.ROBOT_ENEMY_GOAL,
        #     self.goalEnemyColor1,
        #     self.goalEnemyColor2
        # )
    
    def ResetVs(self):
        '''Reset completo: objetos, estado, homografia, buffers e imagens.'''
        self.CreateObjs()

        self._countProcess = 0
        self._count = 0
        self._firstTimeExec = 0
        self.lastMajorTime = 0
        self.currentTime = 0
        self.dT = 0

        self._enemy_last_pos = {}
        self._ally_last_pos = {}

        self.homography_matrix = None
        self.inv_homography_matrix = None
        self.prop_px_cm = 1
        self.min_diag = (7.5 / 4) * np.sqrt(2) * self.prop_px_cm

        self.GPUimg = None
        self.CPUimg = None
        self.emulatorMode = MODE_IMAGE

        self.frameOrigin = None
        self.ballImg = None
        self.fieldReduce = None
        self.frameResult = None

        self.ResetExecutionState()
        
        # 7. Reconstroi o campo
        self.BuildField()

        # 8. Recaptura cores
        self.SetTreeColorDefault()

    def ResetExecutionState(self):
        '''Reseta variáveis temporárias entre frames (preserva robôs/bola/campo/config). Chamar após cada Proc().'''
        self.frameOrigin = None
        self.fieldReduce = None
        self.frameResult = None
        self.imgReduce = None
        self.ballImg = None

        self.binaryObjects = np.zeros((1, 1), dtype=np.uint8)
        self.binaryPlayers = np.zeros((1, 1), dtype=np.uint8)
        self.binaryBall = np.zeros((1, 1), dtype=np.uint8)
        self.binReduceField = np.zeros((1, 1), dtype=np.uint8)
        self.binField = np.zeros((1, 1), dtype=np.uint8)
        self.binaryAllies = np.zeros((1, 1), dtype=np.uint8)

        self.playersWindows = [None, None, None, None, None, None]
        self.alliesWindows = [None, None, None]
        self.enimiesWindows = [None, None, None]

        self.playersCount = 0
        self.alliesCount = 0
        self.enemiesCount = 0
        self.fieldDetectionFailCount  = 0

        # Flags de detecção do frame atual (não mexe no histórico de posições)
        for bot in self.allyTeam:
            bot.detected = False
            bot.possessionBall = False
        for bot in self.enemyTeam:
            bot.detected = False
            bot.possessionBall = False
        if hasattr(self.ball, 'detected'):
            self.ball.detected = False

        if hasattr(self, 'virtual'):
            self.virtualImg = self.virtual.copy()

        self._threads = [t for t in self._threads if t.is_alive()]

        self.colorTree.clear()
        self.SetTreeColorDefault()

    def GetObjects(self):
        '''Retorna dict com aliados, inimigos, bola, campo e timestamp.'''
        objects = {
            ID_Objects.ALLIES:self.allyTeam,
            ID_Objects.ENEMIES: self.enemyTeam,
            ID_Objects.BALL:self.ball,
            ID_Objects.FIELD:self.field,
            'timestamp': self.dT 
        }

        return objects 
    
    def GetFrameProtobuff(self):
        '''Gera o pacote Protobuf (Frame) com dados filtrados pelo Kalman, convertendo cm->m e rodas->velocidade global.'''
        frame = common_pb2.Frame()

        # --- Bola: posição/velocidade filtradas (cm -> m) ---
        if self.ball is not None:
            try:
                if hasattr(self.ball, 'position_filtered') and self.ball.position_filtered is not None:
                    bx, by = self.ball.position_filtered
                else:
                    bx, by = self.ball.position

                bvx, bvy = 0.0, 0.0
                if hasattr(self.ball, 'velocity_filtered') and self.ball.velocity_filtered is not None:
                    vel = self.ball.velocity_filtered
                    bvx, bvy = vel[0], vel[1]

                frame.ball.x = float(bx) / 100.0
                frame.ball.y = float(by) / 100.0
                frame.ball.z = 0.0
                frame.ball.vx = float(bvx) / 100.0
                frame.ball.vy = float(bvy) / 100.0
                frame.ball.vz = 0.0
            except Exception as e:
                if self.debug:
                    print(f"[VisSys][GetFrameProtobuff][ERROR] - Erro ao processar bola: {e}")

        def fill_robot_proto(source_bot, proto_bot):
            # Kalman guarda [x, y, th, vL, vR, w]; protobuf quer [vx, vy] global
            rx, ry = source_bot.position_filtered
            r_theta = source_bot.theta_filtered

            v_wheels = source_bot.velocity_filtered
            vL, vR = v_wheels[0], v_wheels[1]
            r_omega = source_bot.omega_filtered

            v_lin = (vR + vL) / 2.0  # cinemática diferencial -> linear
            r_vx = v_lin * np.cos(r_theta)
            r_vy = v_lin * np.sin(r_theta)

            proto_bot.robot_id = int(source_bot.id.value) if hasattr(source_bot.id, 'value') else int(source_bot.id)
            proto_bot.x = float(rx) / 100.0
            proto_bot.y = float(ry) / 100.0
            proto_bot.orientation = float(r_theta)
            proto_bot.vx = float(r_vx) / 100.0
            proto_bot.vy = float(r_vy) / 100.0
            proto_bot.vorientation = float(r_omega)

        # Aliado -> yellow, Inimigo -> blue
        if self.allyTeam:
            for robot in self.allyTeam:
                if robot is not None and robot.detected:
                    try:
                        robot_pb = frame.robots_yellow.add()
                        fill_robot_proto(robot, robot_pb)
                    except Exception as e:
                        if self.debug:
                            print(f"[VisSys][GetFrameProtobuff][ERROR] - Erro ao processar robô aliado ID {robot.id}: {e}")

        # Time Inimigo -> Blue
        if self.enemyTeam:
            for robot in self.enemyTeam:
                if robot is not None and robot.detected:
                    try:
                        robot_pb = frame.robots_blue.add()
                        fill_robot_proto(robot, robot_pb)
                    except Exception as e:
                        if self.debug:
                            print(f"[VisSys][GetFrameProtobuff][ERROR] - Erro ao processar robô inimigo ID {robot.id}: {e}")

        return frame

    #Puxando as imagens de debug
    def GetDebugImages(self):
        return self.binaryObjects, self.binaryBall, self.binaryPlayers, self.binaryAllies 
        
    def CheckFieldReset(self):
        '''Se houve falhas consecutivas demais na detecção do campo, reseta robôs, bola e homografia.'''
        if self.fieldDetectionFailCount >= self.maxFieldFailures:
            print(f"[VisSys][CheckFieldReset][INFO] - RESET: {self.fieldDetectionFailCount} falhas consecutivas na detecção do campo")
            for bot in (*self.allyTeam, *self.enemyTeam):
                bot.reset()
            if hasattr(self, "ball"):
                self.ball.reset()

            self.homography_matrix = None
            self.inv_homography_matrix = None
            self.fieldDetectedFlag = False
            self.fieldDetectionFailCount = 0

            if self.debug:
                print("[VisSys][CheckFieldReset][DEBUG] - Sistema resetado devido a falhas persistentes na detecção do campo")

    # ==========================================================================================
    # BLOCO 2: PROCESSAMENTO DE IMAGEM, CORES, DESENHO E UTILITÁRIOS
    # ==========================================================================================
    def GrayScale(self,img):
        '''Converte para tons de cinza.'''
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def MedianBlur(self, img, kernelSize = 3):
        '''Filtro de mediana para reduzir ruído.'''
        return cv2.medianBlur(img, kernelSize)

    def BinarizeUp(self, img, threshold=150):
        '''Binariza a imagem (em tons de cinza) por threshold.'''
        _,bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return bin

    def TraitNoise(self, binImg, it=1):
        '''Remove ruído de uma imagem binarizada (abertura + fechamento morfológicos).'''
        try:
            if binImg.dtype != np.uint8:
                binImg = cv2.convertScaleAbs(binImg)
            kernel = self.kernel_rect3
            binImgProc = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel, iterations=it)
            binImgProc = cv2.morphologyEx(binImgProc, cv2.MORPH_CLOSE, kernel, iterations=max(1, it - 1))
            return binImgProc
        except Exception as e:
            print(f"[VisSys][TraitNoise][ERROR] - Erro ao tratar ruído: {e}")
            return binImg
    
    def HighlightImg(self, img, dim=25):
        '''Realça objetos brilhantes em imagem em tons de cinza (top-hat morfológico).'''
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dim, dim))

            # Top-hat: realça áreas mais brilhantes que o entorno
            img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

            # Opcional: uma segunda passada pode suavizar ainda mais o ruído
            img_tophat = cv2.morphologyEx(img_tophat, cv2.MORPH_TOPHAT, kernel)

            # Ajuste de contraste (equivalente a multiplicar brilho)
            img_highlight = cv2.convertScaleAbs(img_tophat, alpha=4.0, beta=0)

            return img_highlight

        except Exception as e:
            print(f"[VisSys][HighlightImg][ERROR] - Erro ao realçar imagem: {e}")
            return img
   
    def ReduceField(self, BinImg, Img, fieldWidth, d=10):
        '''Recorta BinImg/Img para o bounding box do maior contorno (o campo), com margem `d`.'''
        contours, _ = cv2.findContours(BinImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[VisSys][ReduceField][WARNING] - Nenhum contorno encontrado.")
            return BinImg, Img, [0, 0, 0, 0]

        try:
            # Selecionar contornos úteis (descartar ruidos pequenos)
            contours = [c for c in contours if cv2.contourArea(c) > 200]  
            if not contours:
                print("[VisSys][ReduceField][WARNING] - Contornos muito pequenos.")
                return BinImg, Img, [0, 0, 0, 0]

            # Maior contorno
            objT = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(objT)
            cooVetor = [x, y, w, h]

            # Margem segura
            d = max(0, d)

            # Limite dos eixos
            maxH, maxW = BinImg.shape[:2]

            y1 = max(0, y - d)
            y2 = min(maxH, y + h + d)
            x1 = max(0, x - d)
            x2 = min(maxW, x + w + d)

            # Recortes
            bin_Reduce = BinImg[y1:y2, x1:x2]
            img_Reduce = Img[y1:y2, x1:x2]

            # Retângulo do View Capture (ordem consistente)
            pi = np.float32([
                [x1, y1],  # TL
                [x2, y1],  # TR
                [x2, y2],  # BR
                [x1, y2]   # BL
            ])

            rect = Quad(
                Point2D(pi[0, 0], pi[0, 1]),   # TL
                Point2D(pi[1, 0], pi[1, 1]),   # TR
                Point2D(pi[2, 0], pi[2, 1]),   # BR
                Point2D(pi[3, 0], pi[3, 1])    # BL
            )

            # Atualiza viewport
            self.viewCapture.setViewCapture(rect, cooVetor)

        except Exception as e:
            print(f"[VisSys][ReduceField][ERROR] - Erro ao reduzir o campo: {e}")
            return BinImg, Img, [0, 0, 0, 0]

        return bin_Reduce, img_Reduce, cooVetor

    def ConvertMeasures(self, w_cm, w_px):
        '''Atualiza a proporção px/cm (self.prop_px_cm) usada em todo o pipeline.'''
        self.prop_px_cm = w_px / w_cm
        self.min_diag = (7.5 / 4) * np.sqrt(2) * self.prop_px_cm

    def ListPlayers(self, teamList):
        '''Imprime os jogadores de um time (debug).'''
        amount = len(teamList)
        for i in range(amount):
            print(f"[VisSys][ListPlayers][DEBUG] - {teamList[i].team} {teamList[i].id} -> x={teamList[i].pos[0]}, y={teamList[i].pos[1]}")
        print("[VisSys][ListPlayers][DEBUG] - ====================")

    def CreateColorBounds(self, color_array, hue_tolerance=10,
                            saturation_tolerance=50, value_tolerance=50):
        '''
        Cria bandas inferior/superior em HSV a partir de uma cor central.
        Hue usa wrap circular (mod 180): se lower[0] > upper[0], o intervalo cruza o 0°
        -- use MaskInRange() (não cv2.inRange puro) para tratar isso.
        '''
        h = int(color_array[0]) % 180
        s = int(color_array[1])
        v = int(color_array[2])

        low_h = (h - hue_tolerance) % 180
        high_h = (h + hue_tolerance) % 180

        lower_bound = np.array([low_h,  max(0, s - saturation_tolerance), max(0, v - value_tolerance)])
        upper_bound = np.array([high_h, min(255, s + saturation_tolerance), min(255, v + value_tolerance)])

        return lower_bound, upper_bound

    def MaskInRange(self, hsv_img, lower, upper):
        '''
            Equivalente a cv2.inRange, mas tratando o wrap circular do Hue.
            Se lower[0] > upper[0] (intervalo cruza o 0° do matiz), divide em
            dois sub-intervalos [lower_h..179] e [0..upper_h] e retorna o OR.
        '''
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        if lower[0] <= upper[0]:
            return cv2.inRange(hsv_img,
                               lower.astype(np.uint8),
                               upper.astype(np.uint8))

        # Caso wrap: une [lower_h..179] com [0..upper_h] (S e V inalterados)
        lower1 = np.array([lower[0], lower[1], lower[2]], dtype=np.uint8)
        upper1 = np.array([179,      upper[1], upper[2]], dtype=np.uint8)
        lower2 = np.array([0,        lower[1], lower[2]], dtype=np.uint8)
        upper2 = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)

        m1 = cv2.inRange(hsv_img, lower1, upper1)
        m2 = cv2.inRange(hsv_img, lower2, upper2)
        return cv2.bitwise_or(m1, m2)
    
    def SortPoints(self, points):
        '''Ordena 4 pontos como [top_right, top_left, bottom_left, bottom_right].'''
        points = sorted(points, key=lambda p: (p[1], p[0]))
        top_points = sorted(points[:2], key=lambda p: p[0])
        bottom_points = sorted(points[2:], key=lambda p: p[0])
        return np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.int32)

    def SafeCall(self, func, *args, name="", **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[VisSys][SafeCall][ERROR] - Erro ao executar '{name}': {e}")
            traceback.print_exc()
            return None

    # ==========================================================================================
    # BLOCO 3: DESENHO E DEBUG (não afeta a lógica de detecção)
    # ==========================================================================================
    def DrawPlayerCircle(self, imgDegub, robot:Robot):
        '''Desenha o círculo + rótulo (time/ID) de um robô.'''
        xi = int(robot.xi)
        yi = int(robot.yi)
        ri = int(robot.ri)

        if robot.team == ID_Team.TEAM_ALLY:
            team = "A"
            color = (255,0,0)
            if robot.id == ID_Robots.ROBOT_ALLY_GOAL:
                id = "G"
            elif robot.id == ID_Robots.ROBOT_ALLY_1:
                id = "A1"
            elif robot.id == ID_Robots.ROBOT_ALLY_2:
                id = "A2"
        elif robot.team == ID_Team.TEAM_ENEMY:
            team = "E"
            color = (0,0,255)
            if robot.id == ID_Robots.ROBOT_ENEMY_GOAL:
                id = "G"
            elif robot.id == ID_Robots.ROBOT_ENEMY_1:
                id = "A1"
            elif robot.id == ID_Robots.ROBOT_ENEMY_2:
                id = "A2"
        else:
            team = "N/A"
            id = "N/A"
            color = (0,255,0)
        

        cv2.circle(imgDegub, (xi, yi), (ri + 5), color, 2)
        text = f"{team}{id}"
        cv2.putText(imgDegub, text , (int(xi-10),int(yi-ri-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        '''pos =f"({str(xi)},{str(yi)})"
        cv2.putText(imgDegub, pos , (int(xi-30),int(yi+ri+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)'''

    def DrawPlayerVirtual(self, robot:Robot):
        '''
            Desenha um círculo no jogador + seta indicando direção (no debug)
        '''
        xi = int(robot.position[0])
        yi = int(robot.position[1])
        ri = int(robot.radius)

        # converter para dimensões da imagem
        xi, yi = self.GetImageIndice(np.array([xi, yi]))

        # Seleção de cor e rótulo
        if robot.team == ID_Team.TEAM_ALLY:
            team = "A"
            color = (255, 255, 0)
            if robot.id == ID_Robots.ROBOT_ALLY_GOAL:
                id = "G"
            elif robot.id == ID_Robots.ROBOT_ALLY_1:
                id = "A1"
            elif robot.id == ID_Robots.ROBOT_ALLY_2:
                id = "A2"
        elif robot.team == ID_Team.TEAM_ENEMY:
            team = "E"
            color = (0, 0, 255)
            if robot.id == ID_Robots.ROBOT_ENEMY_GOAL:
                id = "G"
            elif robot.id == ID_Robots.ROBOT_ENEMY_1:
                id = "A1"
            elif robot.id == ID_Robots.ROBOT_ENEMY_2:
                id = "A2"
        else:
            team = "N/A"
            id = "N/A"
            color = (0, 255, 0)

        # Desenha o ponto central do robô
        cv2.circle(self.virtualImg, (xi, yi), 4, color, -1)
        text = f"{team}{id}"

        # px/cm = 3  => 6 cm = 18 px
        arrow_len_px = int(6 * 3)

        dir_vec = robot.direction  # já normalizada

        x_end = int(xi + dir_vec[0] * arrow_len_px)
        y_end = int(yi - dir_vec[1] * arrow_len_px)  # Y invertido na imagem

        # Corpo da seta
        cv2.arrowedLine(
                self.virtualImg,
                (xi, yi),
                (x_end, y_end),
                color,
                2,
                tipLength=0.3
            )
        
        # Sem debug → só o círculo e o texto padrão
        cv2.circle(self.virtualImg, (xi, yi), int(3 * robot.radius), color, 1)
        cv2.putText(
                self.virtualImg,
                text,
                (int(xi - 8), int(yi - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

    def DrawFieldDebug(self):
        """Desenha os pontos e linhas do campo no modo debug."""
        self.field.drawPointsField()

        points = [
            self.fieldP1v, self.fieldP2v, self.fieldP3v, self.fieldP4v,
            self.fieldCenterv,
            self.PA1v, self.PA2v, self.PA3v,
            self.PE1v, self.PE2v, self.PE3v,
            self.GA1v, self.GA2v, self.GA3v, self.GA4v,
            self.GAI1v, self.GAI2v, self.GAI3v, self.GAI4v,
            self.GE1v, self.GE2v, self.GE3v, self.GE4v,
            self.GEI1v, self.GEI2v, self.GEI3v, self.GEI4v,
            self.fieldP12v, self.fieldP34v
        ]

        for p in points:
            cv2.circle(self.virtualImg, p, 2, (0, 0, 255), -1)

        # Ponto de referência (O')
        ox, oy = self.xnv, self.ynv
        cv2.circle(self.virtualImg, (ox, oy), 3, (0, 255, 255), -1)
        cv2.putText(self.virtualImg, "O'", (ox + 5, oy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Eixos coordenados (comprimento em pixels)
        arrow_len = 30
        # Eixo X (vermelho) → para a direita
        cv2.arrowedLine(self.virtualImg, (ox, oy), (ox + arrow_len, oy),
                        (0, 0, 255), 1, tipLength=0.2)
        cv2.putText(self.virtualImg, "X", (ox + arrow_len + 3, oy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # Eixo Y (verde) ↑ para cima (lembre: y da imagem é invertido)
        cv2.arrowedLine(self.virtualImg, (ox, oy), (ox, oy - arrow_len),
                        (0, 255, 0), 1, tipLength=0.2)
        cv2.putText(self.virtualImg, "Y", (ox - 15, oy - arrow_len - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
    def DrawFilteredPosition(self, obj, color=(0, 255, 0), label=None):
        """
        Desenha um círculo verde na posição filtrada pelo Kalman (se disponível)
        e um rótulo indicando a qual objeto pertence.
        Se label for None, tenta gerar automaticamente a partir do objeto.
        """
        if self.frameResult is None:
            return
        if not hasattr(obj, 'kalman_initialized') or not obj.kalman_initialized:
            return
        try:
            fx, fy = obj.position_filtered

            # Converte cm -> pixels na imagem original
            xi, yi = self.GetImageRealIndice((fx, fy))

            cv2.circle(self.frameResult, (int(xi), int(yi)), 4, color, -1)

            # Gera label se não fornecido
            if label is None:
                if isinstance(obj, Ball):
                    label = "KF:Bola"
                elif isinstance(obj, Robot):
                    team_prefix = "A" if obj.team == ID_Team.TEAM_ALLY else "E"
                    id_map = {
                        ID_Robots.ROBOT_ALLY_GOAL: "G",
                        ID_Robots.ROBOT_ALLY_1: "1",
                        ID_Robots.ROBOT_ALLY_2: "2",
                        ID_Robots.ROBOT_ENEMY_GOAL: "G",
                        ID_Robots.ROBOT_ENEMY_1: "1",
                        ID_Robots.ROBOT_ENEMY_2: "2",
                    }
                    bot_id_str = id_map.get(obj.id, str(obj.id))
                    label = f"KF:{team_prefix}{bot_id_str}"
                else:
                    label = "KF"

            cv2.putText(self.frameResult, label, (int(xi) - 10, int(yi) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception as e:
            if self.debug:
                print(f"[VisSys][DrawFilteredPosition][ERROR] - {e}")
            
    def DrawKalmanDebug(self, currentTime):
        """
        Desenha, para a bola e para cada robô (aliado/inimigo), o retângulo da ROI
        previsto pelo filtro de Kalman (em cores distintas) e a posição filtrada
        (círculo verde com rótulo). Útil para depuração unificada do Kalman.
        """
        if self.frameResult is None or self.fieldReduce is None:
            return

        img_shape = self.fieldReduce.shape[:2]  # (H, W)

        # Cores e prefixos para cada time
        teams = [
            (self.allyTeam,  (255, 0, 0), "A"),   # aliados: azul
            (self.enemyTeam, (0, 0, 255), "E"),   # inimigos: vermelho
        ]

        # ---- Bola ----
        if getattr(self.ball, "kalman_initialized", False):
            # ROI da bola
            roi_cm = self.SafeCall(self.ball.get_roi, img_shape, t_now=currentTime,
                                name="DrawKalmanDebug(ball).get_roi")
            if roi_cm is not None:
                roi_rect = self.SafeCall(self.GetRoiImg, roi_cm, img_shape,
                                        name="DrawKalmanDebug(ball).GetRoiImg")
                if roi_rect is not None:
                    x, y, w, h = roi_rect
                    cv2.rectangle(self.frameResult, (x, y), (x + w, y + h), (0, 255, 255), 1)
                    cv2.putText(self.frameResult, "K:Bola", (x, y + h + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            # Posição filtrada da bola
            self.DrawFilteredPosition(self.ball)

        # ---- Robôs ----
        for team_list, color, prefix in teams:
            for bot in team_list:
                if not getattr(bot, "kalman_initialized", False):
                    continue

                # ROI do robô
                roi_cm = self.SafeCall(bot.get_roi, image_shape=img_shape, t_now=currentTime,
                                    name=f"DrawKalmanDebug(robot[{bot.id}]).get_roi")
                if roi_cm is not None:
                    roi_rect = self.SafeCall(self.GetRoiImg, roi_cm, img_shape,
                                            name=f"DrawKalmanDebug(robot[{bot.id}]).GetRoiImg")
                    if roi_rect is not None:
                        x, y, w, h = roi_rect
                        cv2.rectangle(self.frameResult, (x, y), (x + w, y + h), color, 1)
                        # Rótulo da ROI
                        id_map = {
                            ID_Robots.ROBOT_ALLY_GOAL: "G",
                            ID_Robots.ROBOT_ALLY_1: "1",
                            ID_Robots.ROBOT_ALLY_2: "2",
                            ID_Robots.ROBOT_ENEMY_GOAL: "G",
                            ID_Robots.ROBOT_ENEMY_1: "1",
                            ID_Robots.ROBOT_ENEMY_2: "2",
                        }
                        bot_id_str = id_map.get(bot.id, str(bot.id))
                        label_roi = f"K:{prefix}{bot_id_str}"
                        cv2.putText(self.frameResult, label_roi, (x, y + h + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                # Posição filtrada do robô
                self.DrawFilteredPosition(bot)
            
    # ==========================================================================================
    # BLOCO 4: PROCESSAMENTO GEOMÉTRICO PURAMENTE (HOMOGRAFIA, COORDENADAS E TAGS)
    # ==========================================================================================

    #função para retornar homografia entre dois sistemas de pontos
    
    def GetHomographyMatrix(self, ptsSrc, ptsFinal):
        '''
        Calcula e salva a homografia 3x3 (imagem real -> imagem virtual, 645x413px, 3px/cm)
        a partir de 4 pontos correspondentes, e sua inversa.
        '''
        ptsSrc = np.array(ptsSrc, dtype='float32')
        ptsFinal = np.array(ptsFinal, dtype='float32')

        self.homography_matrix, _   =   cv2.findHomography(ptsSrc, ptsFinal)

        if self.homography_matrix is not None and np.linalg.cond(self.homography_matrix) < 1 / np.finfo(self.homography_matrix.dtype).eps:
            self.inv_homography_matrix  =   np.linalg.inv(self.homography_matrix)
        else:
            self.inv_homography_matrix = np.eye(3)
            print("[VisSys][GetHomographyMatrix][ERROR] - Matriz de homografia singular, substituída por identidade.")

    def TransformPoint(self, ptSrc):
        '''Aplica a homografia: ponto na imagem real -> ponto na imagem virtual.'''
        ptSrc = np.array([[[ptSrc[0], ptSrc[1]]]], dtype=np.float32)
        ponto_transformado = cv2.perspectiveTransform(ptSrc, self.homography_matrix)
        return ponto_transformado[0][0]

    def InvTransformPoint(self, ptSrc):
        '''Inverso de TransformPoint: imagem virtual -> imagem real.'''
        ptSrc = np.array([[[ptSrc[0], ptSrc[1]]]], dtype=np.float32)
        ponto_transformado = cv2.perspectiveTransform(ptSrc, self.inv_homography_matrix)
        return ponto_transformado[0][0]

    def GetPointVirtual(self, ptSrc):
        '''Converte um ponto (pixel) da imagem virtual para cm no sistema de coordenadas O'.'''
        x = ptSrc[0]
        y = ptSrc[1]
        x_f = (x - self.xnv)/3
        y_f = (self.ynv - y)/3
        return x_f,y_f

    def GetImageIndice(self,ptSrc):
        '''Converte um ponto em cm (O') para pixel na imagem virtual.'''
        x_f =int(ptSrc[0]*3 +self.xnv)
        y_f = int(self.ynv-ptSrc[1]*3)
        return x_f, y_f

    def GetImageRealIndice(self, ptSrc):
        '''Converte um ponto em cm (O') direto para pixel na imagem real (GetImageIndice + InvTransformPoint).'''
        x_i, y_i = self.GetImageIndice(ptSrc)
        return self.InvTransformPoint([x_i, y_i])


    # ==========================================================================================
    # BLOCO 5: LOCALIZAÇÃO GEOMÉTRICA DO ROBÔ (PROTOCOLO DE TAGS T1/T2)
    # ==========================================================================================
    # Implementa o fluxo geométrico fixo descrito no protocolo de tags:
    #   L = 7.5 cm | Tag_Time = L x L/2 (inferior) | Tag_T1 = Tag_T2 = L/2 x L/2 (superior)
    #   Layout: [T1][T2] em cima, [Time] embaixo, centralizado.
    # Usado tanto por DetectBotInRoi quanto por SearchBot, para que as duas vias de
    # busca guiadas pelo Kalman sigam exatamente o mesmo procedimento geométrico.
    TAG_L_CM = 7.5              # L (cm) - FIXO, não modificar
    TAG_ALPHA = np.arctan(1.0 / 3.0)   # ângulo fixo entre o eixo do time e as tags T1/T2
    TAG_WIN = 4                 # meia-janela da amostragem HSV (janela 9x9 => +-4)

    def _RotateVec(self, v, ang):
        ''' rotate(v, ang) conforme passo 4 do protocolo. '''
        c, s = np.cos(ang), np.sin(ang)
        return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])

    def _SampleHsvWindow9(self, roi_hsv_padded, cx, cy, pad=4):
        '''
        Extrai a média HSV de uma janela 9x9 centrada em (cx, cy), a partir de uma
        versão da ROI já com borda replicada (cv2.BORDER_REFLECT, tamanho `pad`),
        garantindo que os índices nunca ultrapassem os limites da imagem original.
        '''
        h_pad, w_pad = roi_hsv_padded.shape[:2]
        px = int(round(cx)) + pad
        py = int(round(cy)) + pad

        x1, x2 = px - pad, px + pad + 1
        y1, y2 = py - pad, py + pad + 1

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_pad, x2), min(h_pad, y2)

        window = roi_hsv_padded[y1:y2, x1:x2]
        if window.size == 0:
            return None
        return window.reshape(-1, 3).mean(axis=0)

    def _FindTeamTagBlobs(self, roi_hsv, team_hsv):
        '''
        Passo 2 (parte 1): localiza os blobs candidatos à Tag_Time (cor team_hsv)
        dentro da ROI, filtrando por área compatível com Tag_Time = L x L/2.
        Retorna lista de contornos ordenada do maior para o menor (área).
        '''
        lower_team, upper_team = self.CreateColorBounds(team_hsv)
        mask_team = self.MaskInRange(roi_hsv, lower_team, upper_team)
        mask_team = cv2.morphologyEx(mask_team, cv2.MORPH_CLOSE, self.struct_ellipse5)
        mask_team = cv2.erode(mask_team, self.struct_ellipse5, iterations=1)

        cnts, _ = cv2.findContours(mask_team, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []

        # Área esperada da Tag_Time (L x L/2), em px^2, com tolerância.
        expected_area_px = (self.TAG_L_CM * self.TAG_L_CM / 2.0) * (self.prop_px_cm ** 2)
        min_area = 0.25 * expected_area_px
        max_area = 4.0 * expected_area_px

        candidates = [c for c in cnts if min_area <= cv2.contourArea(c) <= max_area]
        candidates.sort(key=cv2.contourArea, reverse=True)
        return candidates

    def _BlobCentroidAndTheta(self, cnt):
        ''' Passo 2 (parte 2): centroide C e ângulo de orientação theta do blob. '''
        if len(cnt) >= 5:
            (cx, cy), (_, _), angle_deg = cv2.fitEllipse(cnt)
            theta = np.deg2rad(angle_deg) - np.pi / 2.0  # fitEllipse mede a partir do eixo Y
        else:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                return None
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            mu20 = M['mu20'] / M['m00']
            mu02 = M['mu02'] / M['m00']
            mu11 = M['mu11'] / M['m00']
            theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))

        return np.array([cx, cy], dtype=float), float(theta)

    def _TestBotHypothesis(self, roi_hsv_padded, C, u, primary_hsv, secondary_hsv, dist):
        '''
        Passos 5-7: testa a hipótese primária (sem inversão) e, se falhar, a
        hipótese secundária (invertida 180°). A ordem é MANDATÓRIA.
        Retorna o vetor `direction` (u ou -u) se alguma hipótese bater, senão None.
        '''
        v1 = self._RotateVec(u, +self.TAG_ALPHA)
        v2 = self._RotateVec(u, -self.TAG_ALPHA)

        # --- Passo 6: hipótese primária ---
        P1 = C + dist * v1
        P2 = C + dist * v2
        hsv_p1 = self._SampleHsvWindow9(roi_hsv_padded, P1[0], P1[1])
        hsv_p2 = self._SampleHsvWindow9(roi_hsv_padded, P2[0], P2[1])

        if hsv_p1 is not None and hsv_p2 is not None:
            if self.IsColorMatch(hsv_p1, primary_hsv) and self.IsColorMatch(hsv_p2, secondary_hsv):
                return u

        # --- Passo 7: hipótese secundária (inversão 180°) ---
        v1_opp, v2_opp = -v1, -v2
        P1o = C + dist * v1_opp
        P2o = C + dist * v2_opp
        hsv_p1o = self._SampleHsvWindow9(roi_hsv_padded, P1o[0], P1o[1])
        hsv_p2o = self._SampleHsvWindow9(roi_hsv_padded, P2o[0], P2o[1])

        if hsv_p1o is not None and hsv_p2o is not None:
            if self.IsColorMatch(hsv_p1o, primary_hsv) and self.IsColorMatch(hsv_p2o, secondary_hsv):
                return -u

        return None

    def LocateBotGeometric(self, roi_hsv, team_hsv, primary_hsv, secondary_hsv):
        '''
        Executa o fluxo geométrico completo (passos 1-9 do protocolo) para localizar
        um robô específico (identificado por team_hsv + primary_hsv + secondary_hsv)
        dentro de uma ROI já recortada (roi_hsv).

        Retorna dict {'center': (x,y) local à roi_hsv, 'angle': theta, 'direction':
        (ux,uy), 'contour': contorno da Tag_Time usado} ou None se não encontrado
        em nenhum blob candidato de tamanho adequado.
        '''
        if roi_hsv is None or roi_hsv.size == 0:
            return None

        # Passo 1: L em pixels
        L_px = self.TAG_L_CM * self.prop_px_cm
        # Passo 3: parâmetros geométricos fixos
        dist = L_px * np.sqrt(5) / 4.0

        # Passo 2: candidatos a Tag_Time, do maior para o menor blob
        blobs = self._FindTeamTagBlobs(roi_hsv, team_hsv)
        if not blobs:
            return None

        # Janela replicada uma única vez por ROI (evita recomputar por amostra).
        roi_hsv_padded = cv2.copyMakeBorder(
            roi_hsv, self.TAG_WIN, self.TAG_WIN, self.TAG_WIN, self.TAG_WIN, cv2.BORDER_REFLECT
        )

        # Passo 5-7 (com fallback do passo "REGRA DE VALIDAÇÃO": testar outro blob)
        for cnt in blobs:
            centroid_theta = self._BlobCentroidAndTheta(cnt)
            if centroid_theta is None:
                continue
            C, theta = centroid_theta

            u = np.array([np.cos(theta), np.sin(theta)])
            if u[0] < 0:
                u = -u

            direction = self._TestBotHypothesis(roi_hsv_padded, C, u, primary_hsv, secondary_hsv, dist)
            if direction is None:
                continue  # tenta o próximo blob da cor do time (regra de validação)

            # Passo 8: centro final do robô
            robot_center = C + (L_px / 2.0) * direction
            # Passo 9: ângulo final
            angle = float(np.arctan2(direction[1], direction[0]))

            return {
                'center': robot_center,
                'angle': angle,
                'direction': direction,
                'contour': cnt,
            }

        return None

    def GetCentersColors(self, xci, yci, xmci, ymci, tol=45):
        '''
            Retorna os centros das cores primária e secundária.
            (xci, yci) são os centros do objeot (coordenadas da imagem)
            (xmci, ymci) são os centros da cor principal (coordenadas da image)
            tol = tolerancia da aquisição
        '''
        
        dx = xci - xmci
        dy = yci - ymci
        norm = (dx*dx + dy*dy)**0.5
        if norm < 1e-6:
            return (xci, yci), (xci, yci), (0.0, 0.0)

        dirx = dx / norm
        diry = dy / norm

        # Constantes pré-computadas
        L_m = 2.651650429449553
        L_s = 5.303300858899106

        tol_ang = 1 + (tol + 20) / 100.0
        tol_lin = 1 + tol / 100.0

        theta = np.arctan2(L_m, L_s) * tol_ang
        k = (L_m*L_m + L_s*L_s)**0.5 * tol_lin

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotação manual
        r1x = k * (dirx*cos_t + diry*sin_t)
        r1y = k * (-dirx*sin_t + diry*cos_t)

        r2x = k * (dirx*cos_t - diry*sin_t)
        r2y = k * (dirx*sin_t + diry*cos_t)

        return (xci + r1x, yci + r1y), (xci + r2x, yci + r2y), (dirx, diry)

    def GetHsvMean(self, img_hsv, x, y, kernel=2):
        """
        Retorna a média HSV de uma região quadrada (ex: 3x3) centrada em (x, y).
        kernel=1 → janela 3x3
        kernel=2 → janela 5x5
        """
        h, w = img_hsv.shape[:2]
        x, y = int(x), int(y)
        
        # limites seguros (cortando nas bordas)
        x1, x2 = max(0, x - kernel), min(w, x + kernel + 1)
        y1, y2 = max(0, y - kernel), min(h, y + kernel + 1)
        
        region = img_hsv[y1:y2, x1:x2]
        if region.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)

        mean_hsv = region.mean(axis=(0, 1))

        return mean_hsv
    
    def Hsv2Bgr(self, color_hsv):
        '''Converte uma cor HSV para BGR.'''
        hsv_pixel = np.uint8([[color_hsv]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in bgr_pixel[0,0])

    def ColorInRange(self, hsv_values, lower, upper):
        '''Verifica se cor(es) HSV [Nx3 ou N] estão na faixa [lower,upper], tratando wrap do Hue.'''
        hsv_values = np.atleast_2d(hsv_values).astype(np.uint16)
        lower = np.array(lower, dtype=np.uint16)
        upper = np.array(upper, dtype=np.uint16)

        if lower[0] > upper[0]:  # faixa cruza o 0° do Hue
            mask_hue = ((hsv_values[:, 0] >= lower[0]) | (hsv_values[:, 0] <= upper[0]))
        else:
            mask_hue = ((hsv_values[:, 0] >= lower[0]) & (hsv_values[:, 0] <= upper[0]))

        mask_sat = ((hsv_values[:, 1] >= lower[1]) & (hsv_values[:, 1] <= upper[1]))
        mask_val = ((hsv_values[:, 2] >= lower[2]) & (hsv_values[:, 2] <= upper[2]))
        mask = mask_hue & mask_sat & mask_val

        return mask[0] if hsv_values.shape[0] == 1 else mask

    def IsColorMatch(self, measured_hsv, target_hsv):
        lower, upper = self.CreateColorBounds(target_hsv)
        return self.ColorInRange(measured_hsv, lower, upper)

    # ==========================================================================================
    # BLOCO 6: FUNÇÕES PRINCIPAIS DE DETECÇÃO (Campo, Bola e Jogadores)
    # ==========================================================================================
    def DetectField(self, img, debug):
        """
        Detecta o campo com no máximo DUAS tentativas.
        Retorna o tamanho do campo detectado em centímetros ou -1 se falhar.
        """

        if img is None:
            print("[VisSys][DetectField][ERROR] - A imagem é nula!!")
            self.fieldDetectionFailCount += 1
            self.CheckFieldReset()
            return -1

        h, w = img.shape[:2]
        self.pixelWidth = min(w, h)

        # Vamos tentar apenas com esses dois offsets:
        tentativa_offsets = [self.offSetErode, self.offSetErode + 1]

        campo_detectado = False
        resultado_dp_cm = -1

        for local_offset in tentativa_offsets:
            try:
                # imagem base desta tentativa
                self.frameOrigin = img.copy()

                # pipeline de processamento
                gray = self.GrayScale(self.frameOrigin)
                blur = self.MedianBlur(gray, 3)
                imgProc = self.HighlightImg(blur, self.dimMatrix)
                binary = self.BinarizeUp(imgProc, self.Thrashhold)

                self.binaryObjects = self.TraitNoise(binary, local_offset)

                self.binReduceField, self.fieldReduce, coorVetor = self.ReduceField(
                    self.binaryObjects,
                    self.frameOrigin,
                    self.fieldWidth,
                    self.offSetWindow
                )

                if self.fieldReduce is None:
                    self.fieldReduce = self.frameOrigin.copy()

                contours, _ = cv2.findContours(
                    self.binReduceField,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                encontrou_retangulo = False

                for contour in contours:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx) != 4:
                        continue

                    rectVer = np.array([a[0] for a in approx], dtype=np.int32)
                    rectVer = self.SortPoints(rectVer)

                    top = np.linalg.norm(rectVer[1] - rectVer[0])
                    bottom = np.linalg.norm(rectVer[2] - rectVer[3])
                    left = np.linalg.norm(rectVer[3] - rectVer[0])
                    right = np.linalg.norm(rectVer[2] - rectVer[1])

                    width_px = (top + bottom) / 2
                    height_px = (left + right) / 2

                    self.ConvertMeasures(self.fieldWidth, width_px)
                    modDpCm = width_px / self.prop_px_cm

                    if modDpCm < 60:
                        continue

                    P1, P2, P3, P4 = [Point2D(v[0], v[1]) for v in rectVer]
                    rect = Quad(P1=P1, P2=P2, P3=P3, P4=P4)

                    self.field.updatePos(rect, self.fieldWidth, self.fieldHeight)

                    ptsSource = np.array([P1.getPos(), P2.getPos(), P3.getPos(), P4.getPos()])
                    ptsFinal = np.array([
                        self.fieldP1v, self.fieldP2v,
                        self.fieldP3v, self.fieldP4v
                    ])

                    self.GetHomographyMatrix(ptsSrc=ptsSource, ptsFinal=ptsFinal)

                    self.field.setHomographyMatrix(
                        mHomography=self.homography_matrix,
                        invHomo=self.inv_homography_matrix
                    )

                    encontrou_retangulo = True
                    campo_detectado = True
                    resultado_dp_cm = modDpCm

                    # Resetar contador de falhas
                    self.fieldDetectionFailCount = 0
                    break

                # Se achou retângulo, não tenta mais
                if encontrou_retangulo:
                    self.offSetErode = 0
                    break

            except Exception as e:
                print(f"[VisSys][DetectField][ERROR] - Erro durante detecção: {e}")
                traceback.print_exc()
                continue  # vai para segunda tentativa

        # FIM DAS DUAS TENTATIVAS

        if not campo_detectado:
            self.fieldDetectionFailCount += 1
            if debug:
                print(f"[VisSys][DetectField][DEBUG] - Falha #{self.fieldDetectionFailCount}")
        else:
            self.fieldDetectionFailCount = 0

        self.CheckFieldReset()

        self.frameResult = (self.fieldReduce.copy()
                            if self.fieldReduce is not None
                            else img.copy())

        # ---------------------------------------------------------
        ## Atualização dos Helpers:
        self.winSize = int(18 * self.prop_px_cm)
        self.half_win = self.winSize // 2
        self.playerRadius = 5.3033 * self.prop_px_cm
        self.mainColorRadius = 4.1926 * self.prop_px_cm
        self.single_area = 56.25 * (self.prop_px_cm ** 2)
        self.NOISE_RADIUS_MIN = 0.2 * self.playerRadius

        return resultado_dp_cm if campo_detectado else -1

    #Detectar a imagem da bola na imagem
    def DetectBall(self, img, timestamp, dbg=False, isT=False, hsv_img=None):
        '''
            Função responsável por detectar a bola na imagem

            Necessário informar a imagem que irá ser processada para encontrar a bola. 
            A cor da bola e se irá querer exibir ela na imagem, que tem que ser informada em HSV
        '''
        # 1. Usa o HSV Global
        if hsv_img is not None:
            imgHSV = hsv_img
        else:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if self.fieldReduce is None:
            print("[VisSys][DetectBall][WARNING] - fieldReduce é None")
            self.ballImg = None 
        else:
            self.ballImg = self.fieldReduce.copy()

        self.binaryBall = self.MaskInRange(imgHSV, self.ball_lower_bound, self.ball_upper_bound)

        # Operações de erosão e fechamento (reutiliza elemento estrutural em cache)
        structuringElement = self.struct_ellipse5
        self.binaryBall = cv2.morphologyEx(self.binaryBall, cv2.MORPH_CLOSE, structuringElement)
        self.binaryBall = cv2.erode(self.binaryBall, structuringElement, iterations=1)

        # Encontrando contornos da bola
        contours, _ = cv2.findContours(self.binaryBall, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            ballContour = max(contours, key=cv2.contourArea)
            (xb, yb), rb = cv2.minEnclosingCircle(ballContour)

            # Passando o ponto para o espaço virtual
            xv, yv = self.TransformPoint(np.array([xb, yb]))

            # Valor do objeto dentro do sistema de coordenadas O'
            xcm, ycm = self.GetPointVirtual(np.array([xv, yv]))

            # Tempo que se passou
            time = timestamp

            rb = self.ballRadiusP  # cm -> valor padrão

            # ---------------------------------------------------------
            # Persistência do filtro de Kalman:
            # Se o filtro ainda não foi inicializado, inicialize com setPosition.
            # Caso contrário, apenas atualize o filtro com updatePosition.
            # ---------------------------------------------------------
            if not self.ball.kalman_initialized:
                self.ball.setPosition(xcm, ycm, rb, time)
            else:
                self.ball.updatePosition(xcm, ycm, rb, time)

            self.ball.setImgPosition(xb, yb, rb)
            self.ball.status = True

            rb = int(rb / self.prop_px_cm)
            xb = int(xb)
            yb = int(yb)

            # Circulando bola no frame de debug (detecção em vermelho)
            cv2.circle(self.frameResult, (xb, yb), (rb + 2), (0, 0, 255), 2)
            cv2.putText(self.frameResult, "B", (xb, yb - rb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Desenho da posição filtrada (Kalman) em verde – apenas com debug
            if self.debug and self.ball.kalman_initialized:
                self.DrawFilteredPosition(self.ball)

            # Plotando na imagem virtual
            xv = int(xv)
            yv = int(yv)

            # Desenhando na imagem virtual
            cv2.circle(self.virtualImg, (xv, yv), 4, (255, 255, 255), -1)
            cv2.putText(self.virtualImg, "B", (int(xv - 5), int(yv - rb - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.arrowedLine(self.virtualImg, (xv, yv), 
                            ((xv + int(self.ball.direction[0])), (yv + int(self.ball.direction[1]))), 
                            (0, 255, 255), 2)
        else:
            self.ball.status = False

    def DetectPlayers(self, img, timestamp, dbg=False, isT=False, hsv_img=None):
        '''
        Detecta robôs no frame inteiro. Orquestra:
          PARTE 1 -> DetectPlayerCandidates   (gera candidatos "crus")
          PARTE 2 -> ResolveCollisionBlob     (separa blobs fundidos, chamada pela Parte 1)
          PARTE 3 -> AssociatePlayerCandidates (identifica o ID de cada candidato + Kalman)
        '''
        imgHSV = hsv_img if hsv_img is not None else cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        debug = dbg

        # caches das ultimas posições, para garantir que possa verificar continuidade.
        # Obs: Salvo engano, as classes dos robôs já tem essa informação, talvez seja desnecessário.
        if not hasattr(self, "_enemy_last_pos"):
            self._enemy_last_pos = {}   # {slot: (xcm, ycm)}
        if not hasattr(self, "_ally_last_pos"):
            self._ally_last_pos = {}    # {bot_id: (xcm, ycm)}

        # Inicializo contabilização
        self.playersCount = self.enemiesCount = self.alliesCount = 0
        for bot in (*self.enemyTeam, *self.allyTeam):
            bot.setStatus(False)

        ## Instancio as máscaras binárias como nulas para reiniciar o código.
        # Talvez possa otimizar passando para ele copiar?
        self.binaryPlayers = np.zeros(img.shape[:2], dtype=np.uint8)
        self.binaryAllies = np.zeros(img.shape[:2], dtype=np.uint8)

        # offset (0,0): `img` já é o referencial global (campo inteiro)
        ally_candidates, enemy_candidates = self.DetectPlayerCandidates(
            img, imgHSV, timestamp, x_offset=0, y_offset=0, debug=debug
        )

        self.AssociatePlayerCandidates(ally_candidates, enemy_candidates, imgHSV, timestamp, debug=debug)

        self._countProcess += 1

    # ==========================================================================================
    # PARTE 1/3 -- MOTOR DE DETECÇÃO DE CANDIDATOS (reutilizável em janelas de qualquer tamanho)
    # ==========================================================================================
    def DetectPlayerCandidates(self, img, imgHSV, timestamp, x_offset=0, y_offset=0,
                                        debug=False, max_players=6):
        """
        Gera a máscara genérica de objetos, extrai os contornos "crus" da
        janela recebida e, para cada um, classifica por tamanho: do tamanho
        de 1 robô -> candidato único (via _BuildPlayerCandidate); maior ->
        blob fundido (colisão), delegado para ResolveCollisionBlob (PARTE 2).
        """

        if self.var_d: print("[VS][DEBUG][DETECPLAYER]:  Iniciando fichario da análise.")

        #Etapa 1 - Gero uma máscara para puxar os objetos
        obj_mask = cv2.inRange(imgHSV, self.objectsDarkColor, self.objectsLightColor)

        # Etapa 2 - Exclusão da máscara da bola
        if self.ball.status:
            if self.var_d: print("[VS][DEBUG][DETECPLAYER]:  Bola já foi detectada.")
            xb, yb = int(self.ball.xb) - x_offset, int(self.ball.yb) - y_offset
            r = 5
            h, w = obj_mask.shape[:2]
            y1b, y2b = max(0, yb - r), min(h, yb + r)
            x1b, x2b = max(0, xb - r), min(w, xb + r)
            obj_mask[y1b:y2b, x1b:x2b] = 0

        # =====> Etapa 3 - Morfologia adaptativa por faixas de resolução =====
        px_cm = self.prop_px_cm

        # Seleção dos kernels conforme a resolução
        if px_cm <= 3.0:
            kernel_erode = self.struct_ellipse3
            kernel_close = self.struct_rect5
            it_erode = 2
            faixa = "1/3 (px/cm <= 3.0)"
        elif px_cm <= 4.0:
            kernel_erode = self.struct_ellipse5
            kernel_close = self.struct_rect5
            it_erode = 1
            faixa = "2/3 (3.0 < px/cm <= 4.0)"
        else:
            kernel_erode = self.struct_ellipse5
            kernel_close = self.struct_rect11
            it_erode = 2
            faixa = "3/3 (px/cm > 4.0)"

        # Aplica a morfologia (uma única vez)
        closed_mask = cv2.erode(obj_mask, kernel_erode, iterations=it_erode)
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        if self.var_d:
            print(f"[VS][DEBUG][DETECPLAYER]: Faixa {faixa} - px/cm atual: {px_cm:.2f} - Erosão: {kernel_erode.shape[0]}x{kernel_erode.shape[1]}, Fechamento: {kernel_close.shape[0]}x{kernel_close.shape[1]}")
            
            #cv2.imshow("Mascara dos objetos", obj_mask)
            #cv2.imshow("Mascara dos objetos fechados", closed_mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        # ===================================================================
        # ======== HELPERS em pixels
        winSize = self.winSize
        half_win = self.half_win
        playerRadius = self.playerRadius
        mainColorRadius = self.mainColorRadius
        single_area = self.single_area
        NOISE_RADIUS_MIN =self.NOISE_RADIUS_MIN

        #===========================================================
        if self.var_d:
            txt = f"[VS][DEBUG][DETECPLAYER]: \n - px/cm= {self.prop_px_cm:.2f} \n - Largura da Janela (18cm) = {winSize:.2f} px \n - Raio do Player Previsto = {playerRadius:.2f} px \n - Raio da cor Principal = {mainColorRadius} px \n - Area unitaria de um player = {single_area:.2f} px²"
            print(txt)

        ally_candidates = []
        enemy_candidates = []

        # Etapa 4 - Busca os contornos
        raw_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtra por área mínima (20% da área de um robô)
        min_area_threshold = 0.2 * single_area
        filtered_contours = [cnt for cnt in raw_contours if cv2.contourArea(cnt) >= min_area_threshold]

        if self.var_d:
            print(f"[VS][DEBUG][DETECPLAYER]:  Contornos brutos: {len(raw_contours)}, filtrados: {len(filtered_contours)}")

        i = 1
        for cnt in filtered_contours:
            if self.var_d: print(f"[VS][DEBUG][DETECPLAYER]: ============> Avaliando contorno {i} <=========")
            if self.playersCount >= max_players:
                if self.var_d: print("[VS][DEBUG][DETECPLAYER]: Avaliou todos os players. Saindo do for.")
                break

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < NOISE_RADIUS_MIN:
                continue

            if self.binaryPlayers is not None:
                cv2.drawContours(self.binaryPlayers, [cnt], -1, 255, -1, offset=(x_offset, y_offset))

            area = cv2.contourArea(cnt)
            if area <= 0:
                area = np.pi * r * r

            ratio = area / single_area
            if self.var_d: print(f"[VS][DEBUG][DETECPLAYER]: Avaliando área do contorno {area:.2f}")
            if ratio < 1.2:
                n_est = 1
            else:
                CORRECTION_FACTOR = 1.2
                n_est = int(np.ceil(ratio * CORRECTION_FACTOR))
                n_est = min(n_est, max_players - self.playersCount)

            if self.var_d: print(f"[VS][DEBUG][DETECPLAYER]: Razão = {ratio:.2f} estima ter {n_est:.2f} players nesse grande BLOB")

            if debug and self.frameResult is not None:
                color_circle = (0, 165, 255) if n_est > 1 else (0, 255, 0)
                cv2.circle(self.frameResult, (int(cx) + x_offset, int(cy) + y_offset), int(r) + 5, color_circle, 2)

            if n_est <= 1:
                if self.var_d: print(f"[VS][DEBUG][DETECPLAYER]: Blob de um único indivíduo")
                result = self._BuildPlayerCandidate(
                    img, imgHSV, cx, cy, half_win, playerRadius, mainColorRadius,
                    x_offset, y_offset, cnt=cnt
                )
                if result is not None:
                    if self.var_d: print("[VS][DEBUG][DETECPLAYER]:  Etapa do processamento da colisão realizada.")
                    team_is_enemy, cand = result
                    (enemy_candidates if team_is_enemy else ally_candidates).append(cand)
                    self.playersCount += 1
            else:
                if self.var_d: print(f"[VS][DEBUG][DETECPLAYER]: Blob contém mais de um player.")
                winSizeBlob = 2 * r
                centers, individual_masks = self.ResolveCollisionBlob(
                    img.shape, cnt, cx, cy, r, n_est, winSizeBlob, timestamp,
                    x_offset=x_offset, y_offset=y_offset, imgHSV=imgHSV,
                    img=img
                )
                for idx, (cx_i, cy_i) in enumerate(centers[:n_est]):
                    mask_i = individual_masks[idx] if idx < len(individual_masks) else None
                    result = self._BuildPlayerCandidate(
                        img, imgHSV, cx_i, cy_i, half_win, playerRadius, mainColorRadius,
                        x_offset, y_offset, cnt=None,
                        isBlob=True,
                        individual_mask=mask_i
                    )
                    if result is not None:
                        team_is_enemy, cand = result
                        (enemy_candidates if team_is_enemy else ally_candidates).append(cand)
                        self.playersCount += 1
                        if self.playersCount >= max_players:
                            break
            i += 1

        return ally_candidates, enemy_candidates

    def _BuildPlayerCandidate(self, img, imgHSV, cx, cy, half_win, playerRadius, mainColorRadius, x_offset=0, y_offset=0, cnt=None, isBlob=False, individual_mask=None):
        """
        Helper privado da PARTE 1. Constrói um candidato a robô a partir de um
        centro estimado (cx, cy) em coordenadas LOCAIS a `img`/`imgHSV`.

        `x_offset`/`y_offset` (posição da janela no referencial global) são
        somados apenas nos campos que saem "para fora" desta janela (xi, yi,
        x_m, y_m e a conversão px->cm via TransformPoint) -- o recorte
        "windowActual" continua local, pois só é usado para achar a cor
        dentro da própria janela.

        Se `isBlob=True` e `individual_mask` for fornecido (full-frame), a máscara
        é aplicada sobre a janela HSV para isolar o robô atual (evitando interferência
        do robô vizinho em colisões).
        """
        # Etapa 1 - Define os cantos da janela com base no centro e no raio da janela (half_win)
        # Garante que a janela não ultrapasse as bordas da imagem original.
        x1 = max(0, int(cx - half_win))
        y1 = max(0, int(cy - half_win))
        x2 = min(img.shape[1], int(cx + half_win))
        y2 = min(img.shape[0], int(cy + half_win))

        # Etapa 2 - Recorta a região da imagem RGB correspondente à janela definida.
        # Esta janela será usada posteriormente para exibição e debug.
        windowActual = img[y1:y2, x1:x2]
        if windowActual.size == 0:
            return None

        if self.var_d: 
            print("[VS][DEBUG][DETECPLAYER][BuildPlayersCandidate]:  Janela de tratamento da colisão")

        # Etapa 3 - Recorta a imagem HSV na mesma região.
        # Se for um blob de colisão e tivermos uma máscara individual, aplicamos ela AGORA.
        if self.var_d: 
            print("[VS][DEBUG][DETECPLAYER][BuildPlayersCandidate]:  Etapa de identificação dos players na janela do Blob")
        
        hsv = imgHSV[y1:y2, x1:x2]  # Recorte local da janela

        # Etapa 3.1 - (NOVO) Aplica a máscara individual (se fornecida) para isolar APENAS o robô atual
        if isBlob and individual_mask is not None:
            # A máscara individual está em coordenadas full-frame; recortamos a parte da janela
            mask_local = individual_mask[y1:y2, x1:x2]
            # Aplica a máscara sobre a imagem HSV (zera os pixels fora do robô atual)
            hsv = cv2.bitwise_and(hsv, hsv, mask=mask_local)
            if self.var_d:
                print("[VS][DEBUG][DETECPLAYER][BuildPlayersCandidate]:  Máscara individual aplicada na janela HSV (isolando o robô).")

        # Etapa 4 - Gera máscaras binárias para aliados e inimigos (agora filtradas pela máscara individual, se houver)
        mask_ally = self.MaskInRange(hsv, self.ally_lower_bound, self.ally_upper_bound)
        mask_enemy = self.MaskInRange(hsv, self.enemy_lower_bound, self.enemy_upper_bound)

        # Etapa 5 - Calcula a área (número de pixels) de cada máscara e decide o time
        # com base na proporção. Se nenhuma cor for predominante (>40%), descarta o candidato.
        ally_area = cv2.countNonZero(mask_ally)
        enemy_area = cv2.countNonZero(mask_enemy)
        total_area = max(ally_area + enemy_area, 1)
        ally_ratio = ally_area / total_area
        enemy_ratio = enemy_area / total_area
        
        if ally_area == 0 and enemy_area == 0:
            return None
        if ally_area >= enemy_area:
            if ally_ratio < 0.4:
                return None
            team_is_enemy = False
        else:
            if enemy_ratio < 0.4:
                return None
            team_is_enemy = True

        # Etapa 6 - Seleciona a máscara correspondente ao time definido e encontra
        # o maior contorno externo. Em seguida, calcula o círculo mínimo envolvente
        # para obter o centro (x_m, y_m) e o raio rc da mancha de cor.
        curr_mask = mask_enemy if team_is_enemy else mask_ally
        contour = max(cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                    key=cv2.contourArea, default=None)
        if contour is None:
            return None
        (x_m, y_m), rc = cv2.minEnclosingCircle(contour)

        # Etapa 7 - Converte as coordenadas do centro da mancha para o sistema
        # da imagem inteira (somando x1, y1) e verifica se o raio da mancha é
        # grande o suficiente para ser considerado um robô (limiar diferente para
        # inimigos e aliados).
        x_m += x1
        y_m += y1
        min_rc = (0.75 if team_is_enemy else 0.5) * mainColorRadius
        if rc < min_rc:
            return None

        # Etapa 8 - Calcula a direção (vetor unitário) que aponta do centro da
        # mancha de cor (x_m, y_m) para o centro estimado (cx, cy). A inversão
        # do eixo y (uso de -cy e -y_m) é para adequar ao sistema de coordenadas
        # da imagem (origem no canto superior esquerdo).
        direction = np.array([cx, -cy]) - np.array([x_m, -y_m])
        modDir = np.linalg.norm(direction)
        if modDir > 1e-6:
            direction = direction / modDir

        # Etapa 9 - Soma o offset da janela (x_offset, y_offset) para obter
        # coordenadas globais (no sistema do campo inteiro). Essas coordenadas
        # serão usadas para desenho, identificação na Parte 3 e conversão para cm.
        cx_g, cy_g = cx + x_offset, cy + y_offset
        x_m_g, y_m_g = x_m + x_offset, y_m + y_offset

        xcm, ycm = self.GetPointVirtual(self.TransformPoint(np.array([cx_g, cy_g])))

        # Etapa 10 - Retorna uma tupla (team_is_enemy, dicionário com todas as
        # informações do candidato: centro estimado, centro da mancha, posição em cm,
        # direção, janela recortada e o contorno original (se fornecido)).
        return team_is_enemy, {
            "xi": cx_g, "yi": cy_g, "ri": playerRadius,
            "x_m": x_m_g, "y_m": y_m_g,
            "xcm": xcm, "ycm": ycm, "rcm": 5.30,
            "direction": direction,
            "windowActual": windowActual,
            "contour": cnt,  # pode ser None
        }

    # ==========================================================================================
    # PARTE 2/3 -- TRATAMENTO DE BLOBS GIGANTES (COLISÃO DE 2+ ROBÔS)
    # ==========================================================================================
    def ResolveCollisionBlob(self, img_shape, cnt, cx, cy, r, n_est, winSizeBlob, timestamp,
                            x_offset=0, y_offset=0, imgHSV=None, img=None,
                            save_path="src/data/study/"):
        """
        Resolve um blob fundido (colisão de múltiplos robôs) usando K-Means para
        encontrar os centros individuais.

        Parâmetros:
            img_shape: tupla (altura, largura) da imagem original.
            cnt: contorno do blob (em coordenadas full-frame).
            cx, cy: centro aproximado do blob (full-frame).
            r: raio aproximado do blob (em pixels).
            n_est: número estimado de robôs dentro deste blob.
            winSizeBlob: tamanho da janela (não usado diretamente, mantido para compatibilidade).
            timestamp: timestamp (para identificação única).
            x_offset, y_offset: offset da janela (não usado, mantido para compatibilidade).
            imgHSV: imagem HSV full-frame (opcional, para debug).
            img: imagem RGB full-frame (opcional, para debug).
            save_path: diretório onde salvar as imagens de debug.

        Retorna:
            centers: lista de tuplas (x, y) com os centros encontrados (full-frame).
            individual_masks: lista de máscaras binárias full-frame, uma para cada robô.
        """
        # ==================== INÍCIO DO DEBUG ====================
        if self.var_d:
            print("[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Iniciando tratamento da colisão para encontrar centros.")
            print("[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Estimativa de {} robôs neste blob.".format(n_est))
            print("[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Centro aproximado ({:.1f}, {:.1f}) e Raio {:.1f} px".format(cx, cy, r))

        # Cria o diretório de saída se não existir
        full_save_path = os.path.join(os.getcwd(), save_path)
        os.makedirs(full_save_path, exist_ok=True)

        # Identificador único para este blob
        blob_id = f"blob__cx{int(cx)}_cy{int(cy)}"

        # ==================== ETAPA 1: BOUNDING BOX DO BLOB ====================
        x, y, w, h = cv2.boundingRect(cnt)
        margin = 5
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_shape[1], x + w + margin)
        y2 = min(img_shape[0], y + h + margin)

        if self.var_d:
            print("[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Bounding Box do Blob: ({}, {}) -> ({}, {})".format(x1, y1, x2, y2))

        # ==================== ETAPA 2: MÁSCARA FULL-FRAME DO BLOB ====================
        mask_full = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_full, [cnt], -1, 255, -1)

        # ==================== ETAPA 3: RECORTAR AS ROIS ====================
        mask_roi = mask_full[y1:y2, x1:x2]
        hsv_roi = None
        img_roi = None

        if img is not None:
            img_roi = img[y1:y2, x1:x2]
        if imgHSV is not None:
            hsv_roi = imgHSV[y1:y2, x1:x2]

        # ==================== SALVA IMAGENS DE DEBUG (JÁ EXISTENTES) ====================
        if img_roi is not None and self.var_d:
            rgb_filename = os.path.join(full_save_path, f"{blob_id}_rgb.png")
            cv2.imwrite(rgb_filename, img_roi)
            if self.var_d:
                print(f"[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Imagem RGB salva em {rgb_filename}")

        if self.var_d:
            mask_filename = os.path.join(full_save_path, f"{blob_id}_mask_blob.png")
            cv2.imwrite(mask_filename, mask_roi)
            print(f"[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Máscara do Blob salva em {mask_filename}")

        # ====================================================================
        # ====== NOVO: LOCALIZAÇÃO DOS CENTROS USANDO K-MEANS ======
        # ====================================================================
        if self.var_d:
            print("[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Aplicando K-Means para encontrar os centros dos robôs.")

        # 1. Obter todos os pixels pertencentes ao blob (pontos não-zero da máscara)
        pts = cv2.findNonZero(mask_roi)  # Retorna um array (N, 1, 2) ou None

        if pts is None:
            # Se não houver pontos (máscara vazia), retorna listas vazias
            if self.var_d:
                print("[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  ATENÇÃO: Máscara do blob vazia. Retornando vazio.")
            return [], []

        # 2. Converter para o formato exigido pelo K-Means: (N, 2) com float32
        pts = np.float32(pts).reshape(-1, 2)

        # 3. Definir critérios de parada: 30 iterações ou precisão de 0.1 pixel
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        # 4. Aplicar K-Means
        #    - n_est: número de clusters (robôs)
        #    - tentativas: 10 (para evitar mínimos locais)
        #    - KMEANS_PP_CENTERS: inicialização inteligente (K-Means++)
        _, _, centers_roi = cv2.kmeans(pts, n_est, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # centers_roi é um array (n_est, 2) com as coordenadas (x, y) na ROI

        if self.var_d:
            print(f"[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  K-Means encontrou {len(centers_roi)} centros na ROI.")

        # ==================== CONVERTER CENTROS PARA COORDENADAS GLOBAIS ====================
        centers = []          # Lista de (cx_global, cy_global)
        individual_masks = [] # Lista de máscaras full-frame para cada robô

        for c in centers_roi:
            # c é [x, y] em coordenadas locais da ROI
            cx_global = int(c[0] + x1)
            cy_global = int(c[1] + y1)
            centers.append((cx_global, cy_global))

            # Gera máscara individual (círculo de raio r*0.8 ao redor do centro)
            mask_ind = np.zeros(img_shape[:2], dtype=np.uint8)
            cv2.circle(mask_ind, (cx_global, cy_global), int(r * 0.8), 255, -1)
            individual_masks.append(mask_ind)

            if self.var_d:
                print(f"[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Centro K-Means: global ({cx_global}, {cy_global})")

        # ==================== SALVAR IMAGEM DE DEBUG COM OS CENTROS ====================
        # Criamos uma imagem colorida para debug: se tivermos img_roi, usamos ela,
        # senão, usamos a máscara em escala de cinza convertida para BGR.
        if img_roi is not None:
            debug_img = img_roi.copy()
        else:
            # Converte a máscara para 3 canais (escala de cinza -> BGR)
            debug_img = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)

        # Desenha um círculo vermelho em cada centro encontrado (coordenadas locais)
        for c in centers_roi:
            cx_roi, cy_roi = int(c[0]), int(c[1])
            cv2.circle(debug_img, (cx_roi, cy_roi), int(r * 0.4), (0, 0, 255), 2)  # Vermelho
            # Opcional: coloca um ponto central
            cv2.circle(debug_img, (cx_roi, cy_roi), 2, (0, 0, 255), -1)

        # Salva a imagem com os centros
        centers_filename = os.path.join(full_save_path, f"{blob_id}_centers_kmeans.png")
        cv2.imwrite(centers_filename, debug_img)
        if self.var_d:
            print(f"[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Imagem com centros salva em {centers_filename}")

        # ==================== SALVAR MÁSCARAS INDIVIDUAIS (como antes) ====================
        for i, (cx_g, cy_g) in enumerate(centers):
            mask_ind_roi = individual_masks[i][y1:y2, x1:x2]
            ind_filename = os.path.join(full_save_path, f"{blob_id}_mask_ind_{i+1}.png")
            if self.var_d:
                cv2.imwrite(ind_filename, mask_ind_roi)
                print(f"[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Máscara Individual {i+1} salva em {ind_filename}")

        # ==================== FINALIZAÇÃO ====================
        if self.var_d:
            print("[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Centros retornados: {}".format(centers))
            print("[VS][DEBUG][DETECPLAYER][ResolveCollisionBlob]:  Finalizando tratamento. Todas as imagens foram salvas em {}".format(full_save_path))

        return centers, individual_masks
    # ==========================================================================================
    # PARTE 3/3 -- IDENTIFICAÇÃO DOS CANDIDATOS + ATUALIZAÇÃO DO FILTRO DE KALMAN
    # ==========================================================================================
    def AssociatePlayerCandidates(self, ally_candidates, enemy_candidates, imgHSV, timestamp, debug=False):
        """
        Decide a QUAL robô específico (ID) cada candidato "cru" pertence.
        - Aliados: identificação pela árvore de cores (já populada no início).
        - Inimigos: pareamento por posição na primeira detecção; depois usa a árvore
        de cores como critério adicional para evitar troca de IDs em colisões.
        """
        MAX_JUMP_CM = self.MAX_JUMP_CM

        # ================= Associação de inimigos =====================
        if enemy_candidates:
            # --- Fase 1: pareamento por posição (cold start e fallback) ---
            pairs = []
            n_slots = min(3, len(self.enemyTeam))
            for ci, cand in enumerate(enemy_candidates):
                for slot in range(n_slots):
                    last = self._enemy_last_pos.get(slot)
                    dist = 0.0 if last is None else float(np.hypot(cand["xcm"] - last[0], cand["ycm"] - last[1]))
                    pairs.append((dist, ci, slot))
            pairs.sort(key=lambda p: p[0])
            used_candidates, used_slots = set(), set()
            assignment = {}  # slot -> índice do candidato

            # --- Fase 2: para cada par, tenta associar, usando cor como confirmação, não como barreira ---
            for dist, ci, slot in pairs:
                if ci in used_candidates or slot in used_slots:
                    continue
                last = self._enemy_last_pos.get(slot)
                if last is not None and dist > MAX_JUMP_CM:
                    continue

                cand = enemy_candidates[ci]
                # Obtém as cores do candidato
                Color_p_pt, Color_s_pt, _ = self.GetCentersColors(cand["xi"], cand["yi"], cand["x_m"], cand["y_m"])
                Color_p = self.GetHsvMean(imgHSV, Color_p_pt[0], Color_p_pt[1])
                Color_s = self.GetHsvMean(imgHSV, Color_s_pt[0], Color_s_pt[1])

                # Verifica se existe algum match na árvore para essas cores
                existing = self.colorTree.find_by_colors(self.enemyColor, Color_p, Color_s)
                
                # Decisão:
                # - Se há match e o robot_id é diferente do slot, rejeitamos (evita troca de IDs).
                # - Caso contrário (sem match ou match com o próprio slot), permitimos a associação.
                if existing is not None and existing['robot_id'] != slot:
                    continue

                # Se chegou aqui, a associação é permitida (por posição ou por cor confirmada)
                assignment[slot] = ci
                used_candidates.add(ci)
                used_slots.add(slot)

            # --- Fase 3: aplica as associações e atualiza o Kalman ---
            for slot, ci in assignment.items():
                cand = enemy_candidates[ci]
                bot = self.enemyTeam[slot]
                # Recalcula as cores (já temos, mas garantimos)
                Color_p_pt, Color_s_pt, _ = self.GetCentersColors(cand["xi"], cand["yi"], cand["x_m"], cand["y_m"])
                Color_p = self.GetHsvMean(imgHSV, Color_p_pt[0], Color_p_pt[1])
                Color_s = self.GetHsvMean(imgHSV, Color_s_pt[0], Color_s_pt[1])
                # Atualiza a árvore (se já existir, atualiza a média; se não, insere)
                self.colorTree.add_robot(ID_Team.TEAM_ENEMY, slot, self.enemyColor, Color_p, Color_s)

                if not bot.kalman_initialized:
                    bot.setPosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)
                else:
                    bot.updatePosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)
                bot.updtPositionImg(cand["xi"], cand["yi"], cand["ri"])
                bot.setStatus(True)
                bot.setRadius(cand["rcm"])
                bot.setColor(colorT=self.enemyColor, colorP=Color_p, colorS=Color_s)

                self.DrawPlayerVirtual(bot)
                self._enemy_last_pos[slot] = (cand["xcm"], cand["ycm"])
                self.enemiesCount += 1

                self.DrawPlayerCircle(self.frameResult, bot)
                if debug:
                    if bot.kalman_initialized:
                        self.DrawFilteredPosition(bot)
                    cx2, cy2 = int(Color_p_pt[0]), int(Color_p_pt[1])
                    bgr_p = self.Hsv2Bgr(Color_p)
                    cv2.circle(self.frameResult, (cx2, cy2), 4, (0, 0, 0), -1)
                    cv2.circle(self.frameResult, (cx2, cy2), 3, bgr_p, -1)
                    cx3, cy3 = int(Color_s_pt[0]), int(Color_s_pt[1])
                    bgr_s = self.Hsv2Bgr(Color_s)
                    cv2.circle(self.frameResult, (cx3, cy3), 4, (0, 0, 0), -1)
                    cv2.circle(self.frameResult, (cx3, cy3), 3, bgr_s, -1)

        # ================= Associação de aliados ======================
        # Para aliados, a árvore já está populada desde o início (SetTreeColorDefault),
        # então usamos a identificação por cor sem necessidade de fallback.
        for cand in ally_candidates:
            if self.alliesCount >= 3:
                break

            match, Color_p, Color_s = self.IdentifyCandidateByColor(cand, imgHSV, self.allyColor)
            if match is None:
                continue
            bot_id = match['robot_id']
            bot = self.allyTeam[bot_id]

            # Gating de distância (segurança)
            last = self._ally_last_pos.get(bot_id)
            if last is not None:
                dist = float(np.hypot(cand["xcm"] - last[0], cand["ycm"] - last[1]))
                if dist > MAX_JUMP_CM:
                    continue

            if not bot.kalman_initialized:
                bot.setPosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)
            else:
                bot.updatePosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)
            bot.updtPositionImg(cand["xi"], cand["yi"], cand["ri"])
            bot.setStatus(True)
            bot.setRadius(cand["rcm"])
            bot.setColor(colorT=self.allyColor, colorP=Color_p, colorS=Color_s)

            if cand["contour"] is not None:
                cv2.drawContours(self.binaryAllies, [cand["contour"]], -1, 255, -1)
            else:
                cv2.circle(self.binaryAllies, (int(cand["xi"]), int(cand["yi"])), int(cand["ri"]), 255, -1)

            self.DrawPlayerCircle(self.frameResult, bot)
            if debug:
                if bot.kalman_initialized:
                    self.DrawFilteredPosition(bot)
                cv2.circle(self.frameResult, (int(cand["x_m"]), int(cand["y_m"])), 4, (255, 128, 255), -1)

            self.DrawPlayerVirtual(bot)
            self._ally_last_pos[bot_id] = (cand["xcm"], cand["ycm"])
            self.alliesCount += 1


    def IdentifyCandidateByColor(self, cand, imgHSV, main_color):
        """
        Utilitário de CONSULTA (não altera nada) para a PARTE 3 -- é o ponto
        de extensão citado na docstring de AssociatePlayerCandidates.

        Extrai a cor primária/secundária de um candidato exatamente como já
        é feito para os inimigos dentro de AssociatePlayerCandidates, e
        consulta a árvore de cores (self.colorTree) para descobrir a qual
        time/robô aquela combinação de cores pertence -- uma segunda fonte
        de identidade (por aparência), independente da posição, que é
        justamente o que ajuda a desempatar robôs (aliados OU inimigos) logo
        após uma colisão, quando a posição sozinha não é confiável.

        Parâmetros:
            cand: dict de candidato (formato retornado por
                DetectPlayerCandidates/_BuildPlayerCandidate).
            imgHSV: HSV no mesmo referencial em que `cand` foi construído.
            main_color: cor-time a comparar (self.allyColor ou
                self.enemyColor, conforme o candidato).

        Retorna (match, Color_p, Color_s):
            match: dict {'team':..., 'robot_id':...} retornado por
                self.colorTree.find_by_colors(...), ou None se não achar
                nenhuma cor compatível já cadastrada na árvore.
            Color_p, Color_s: as cores HSV médias extraídas do candidato
                (úteis caso queira, em seguida, salvar/atualizar a árvore
                com self.colorTree.add_robot(...), do mesmo jeito que já é
                feito hoje para os inimigos em AssociatePlayerCandidates).
        """
        Color_p_pt, Color_s_pt, _ = self.GetCentersColors(cand["xi"], cand["yi"], cand["x_m"], cand["y_m"])
        Color_p = self.GetHsvMean(imgHSV, Color_p_pt[0], Color_p_pt[1])
        Color_s = self.GetHsvMean(imgHSV, Color_s_pt[0], Color_s_pt[1])
        match = self.colorTree.find_by_colors(main_color, Color_p, Color_s)
        return match, Color_p, Color_s

    # ==========================================================================================
    # BLOCO 7: DETECÇÃO ADAPTATIVA COM ROI (USANDO KALMAN PARA GUIAR A BUSCA)
    # ==========================================================================================
    
    # Método novo para procurar se existe um robô na janela
    def SearchBots(self, img, timestamp, debug=False) -> list:
        if img is None:
            return []

        H, W = img.shape[:2]
        detected_list = []
        processed_rois = set()  # Evita processar a mesma ROI mais de uma vez

        # Lista de alvos: (ObjetoRobo, EnumTime)
        targets = []
        for b in self.allyTeam:
            targets.append((b, ID_Team.TEAM_ALLY))
        for b in self.enemyTeam:
            targets.append((b, ID_Team.TEAM_ENEMY))

        # Primeiro, tenta a geometria de tags para cada robô
        for bot, team_enum in targets:
            roi_img, roi_rect = self.PredictRobot([H, W], team_enum, bot.id, timestamp)
            if roi_img is None or roi_img.size == 0:
                continue

            # Tenta detectar via geometria de tags
            result = self.DetectBotInRoi(roi_img, roi_rect, bot, debug)
            if result:
                detected_list.extend(result)
                if debug:
                    xr, yr, wr, hr = roi_rect
                    cv2.rectangle(self.frameResult, (xr, yr), (xr+wr, yr+hr), (0, 255, 0), 1)
            else:
                # Se falhou, marca a ROI para processamento via candidatos (colisão)
                roi_key = (roi_rect[0], roi_rect[1], roi_rect[2], roi_rect[3])
                if roi_key not in processed_rois:
                    processed_rois.add(roi_key)
                    # Processa a ROI para detectar todos os robôs ali
                    roi_results = self._ProcessROIForRobots(roi_img, roi_rect, timestamp, debug)
                    if roi_results:
                        detected_list.extend(roi_results)
                        if debug:
                            xr, yr, wr, hr = roi_rect
                            cv2.rectangle(self.frameResult, (xr, yr), (xr+wr, yr+hr), (0, 255, 255), 2)  # amarelo para indicar fallback

        return detected_list

    def DetectBotInRoi(self, roi_img, roi_rect, target_bot, debug=False) -> list:
            """
            Localiza o `target_bot` dentro da ROI prevista pelo Kalman usando o fluxo
            geométrico fixo das tags (Tag_Time + Tag_T1 + Tag_T2). Diferente da versão
            anterior (que buscava candidatos genéricos por tamanho e depois tentava
            identificar o ID via árvore de cores), aqui já sabemos exatamente qual
            robô estamos procurando: usamos as cores do próprio `target_bot`
            (colorTeam / colorCar1 / colorCar2) para achar sua Tag_Time e validar a
            orientação via T1/T2, conforme o protocolo geométrico.
            """
            # 0. Validações básicas
            if roi_img is None or roi_img.size == 0:
                return []

            # Desempacota offsets globais (x0, y0 é o canto sup. esq. do ROI no campo)
            x0, y0, w0, h0 = roi_rect
            results = []

            # Usamos as coordenadas do roi_rect para cortar a matriz HSV global
            roi_hsv = self.imgHSV[y0: y0 + h0, x0: x0 + w0]
            if roi_hsv is None or roi_hsv.size == 0:
                return []

            team_hsv = target_bot.colorTeam
            primary_hsv = target_bot.colorCar1
            secondary_hsv = target_bot.colorCar2
            if team_hsv is None or primary_hsv is None or secondary_hsv is None:
                return []

            # --------------------------------------------------------
            # Fluxo geométrico (passos 1-9 do protocolo de tags)
            # --------------------------------------------------------
            hit = self.LocateBotGeometric(roi_hsv, team_hsv, primary_hsv, secondary_hsv)
            if hit is None:
                return []

            center_local = hit['center']
            direction_img = hit['direction']  # (ux, uy) em coordenadas de imagem (y p/ baixo)
            cnt = hit['contour']

            # Coordenadas GLOBAIS do centro do robô
            xi_global = int(round(center_local[0] + x0))
            yi_global = int(round(center_local[1] + y0))

            # Direção no "padrão do mundo": inverte Y (mesma convenção usada no resto
            # do módulo, ex.: dy = -(yi_global - ym_global)).
            direction = np.array([direction_img[0], -direction_img[1]], dtype=float)
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
            theta = float(np.arctan2(direction[1], direction[0]))

            # Raio de referência do robô (para desenho / gating de tamanho a jusante)
            playerRadius = (self.TAG_L_CM / 2.0) * np.sqrt(2) * self.prop_px_cm
            ri = playerRadius

            # --------------------------------------------------------
            # Recorte da janela do robô (bot_win), igual ao comportamento anterior
            # --------------------------------------------------------
            winSize = int(18 * self.prop_px_cm)
            half_win = winSize // 2
            xi_local, yi_local = center_local[0], center_local[1]
            x1_local = max(0, int(xi_local - half_win))
            y1_local = max(0, int(yi_local - half_win))
            x2_local = min(w0, int(xi_local + half_win))
            y2_local = min(h0, int(yi_local + half_win))
            bot_win = roi_img[y1_local:y2_local, x1_local:x2_local]

            # --------------------------------------------------------
            # Conversão px -> cm e montagem do resultado
            # --------------------------------------------------------
            xv, yv = self.TransformPoint(np.array([xi_global, yi_global]))
            xcm, ycm = self.GetPointVirtual(np.array([xv, yv]))

            result_data = {
                "id": target_bot.id,
                "team": target_bot.team,
                "x": xcm,
                "y": ycm,
                "theta": theta,
                "direction": direction,
                "img_x": xi_global,
                "img_y": yi_global,
                "img_r": ri,
                "bot_win": bot_win,
                "contour_global": None,
            }

            # Contorno global da Tag_Time encontrada (para desenho/depuração)
            c_global = cnt.copy()
            c_global[:, 0, 0] += x0
            c_global[:, 0, 1] += y0
            result_data["contour_global"] = c_global

            if self.binaryPlayers is not None:
                cv2.drawContours(self.binaryPlayers, [c_global], -1, 255, -1)

            results.append(result_data)
            return results

    def SearchBall(self, roi_img, roi_rect, timestamp, debug=False, name=""):
            """
            Busca a bola dentro do recorte (roi_img).
            Retorna coordenadas globais somando o offset (roi_rect).
            """
            # Se a imagem do recorte for inválida
            if roi_img is None or roi_img.size == 0:
                return {'found': False}

            # Desempacota o offset global
            x0, y0, w0, h0 = roi_rect
            
            # 1. Processamento na imagem recortada (rápido)
            # Não precisa recriar wnd, roi_img JÁ É a janela
            hsv = self.imgHSV[y0 : y0 + h0, x0 : x0 + w0]
            
            mask = self.MaskInRange(hsv, self.ball_lower_bound, self.ball_upper_bound)

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.struct_ellipse5)
            mask = cv2.erode(mask, self.struct_ellipse5, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return {'found': False}

            # 2. Encontrar o centro local
            c = max(contours, key=cv2.contourArea)
            (xb_local, yb_local), r_ball_px = cv2.minEnclosingCircle(c)

            # 3. Converter para Global (Fundamental!)
            xb = int(xb_local + x0)
            yb = int(yb_local + y0)

            # Debug Visual Global
            if self.binaryBall is not None:
                cv2.circle(self.binaryBall, (xb, yb), int(r_ball_px * 1.3), 255, -1)
            
            if debug:
                # Aqui desenhamos na imagem global (frameResult) usando as coords globais
                cv2.circle(self.frameResult, (xb, yb), int(r_ball_px) + 2, (0, 0, 255), 2)
                # Opcional: desenhar o retângulo de busca
                cv2.rectangle(self.frameResult, (x0, y0), (x0+w0, y0+h0), (0, 255, 255), 1)

            # 4. Transformação para Mundo Virtual
            xv, yv = self.TransformPoint(np.array([xb, yb]))
            xcm, ycm = self.GetPointVirtual(np.array([xv, yv]))

            return {
                'found': True, 
                'x': xcm, 
                'y': ycm,
                'img_x': xb, 
                'img_y': yb, 
                'img_r': r_ball_px,
                'radius': self.ballRadiusP # Ou usar r_ball_px convertido
            }

    # ==========================================================================================
    # BLOCO 8: PREDIÇÃO E FILTRAGEM COM KALMAN (MÉTODOS DE ROI E FALLBACK)
    # ==========================================================================================

    def GetRoiImg(self, ROI_obj, img_shape, min_size=12, max_ratio=0.5):
        """
        Converte ROI do espaço virtual (cm) para coordenadas da imagem em pixels.
        Garante limites mínimos e máximos.
        """
        x_cm, y_cm, w_cm, h_cm = ROI_obj

        # Canto superior esquerdo e inferior direito
        top_left_real = self.GetImageRealIndice((x_cm, y_cm))
        bottom_right_real = self.GetImageRealIndice((x_cm + w_cm, y_cm + h_cm))

        # ATENÇÃO: getPointVirtual/getImageIndice inverte o eixo Y (mundo cm -> pixel)
        # e a homografia inversa pode rotacionar/espelhar. Portanto NÃO se pode assumir
        # que "top_left_real" continua sendo o canto superior-esquerdo em pixels.
        # Usamos min/max dos dois cantos para obter o canto superior-esquerdo real.
        cx0, cy0 = int(top_left_real[0]), int(top_left_real[1])
        cx1, cy1 = int(bottom_right_real[0]), int(bottom_right_real[1])

        tl_x, br_x = min(cx0, cx1), max(cx0, cx1)
        tl_y, br_y = min(cy0, cy1), max(cy0, cy1)

        # Ajuste dentro da imagem
        H, W = img_shape[:2]
        tl_x = max(0, min(tl_x, W-1))
        tl_y = max(0, min(tl_y, H-1))
        br_x = max(0, min(br_x, W))
        br_y = max(0, min(br_y, H))

        # Dimensões
        w_img = max(br_x - tl_x, min_size)
        h_img = max(br_y - tl_y, min_size)

        # Limite máximo relativo à imagem
        max_w = int(W * max_ratio)
        max_h = int(H * max_ratio)
        w_img = min(w_img, max_w)
        h_img = min(h_img, max_h)

        return (tl_x, tl_y, w_img, h_img)

    def PredictBall(self, img_shape, timestamp):
            """
            Retorna (imagem_crop, (x, y, w, h)).
            """
            h_img, w_img = img_shape[:2]

            # 1. Obter ROI virtual (cm) do Kalman
            roi_virtual_cm = self.SafeCall(
                self.ball.get_roi,
                img_shape,
                t_now=timestamp,
                name="ball.get_roi"
            )

            fallback_rect = (0, 0, int(w_img*0.5), int(h_img*0.5))

            if roi_virtual_cm is None:
                # Se falhar, tenta pegar o centro da imagem
                return None, fallback_rect

            # 2. Converter para Pixels (Tupla x,y,w,h)
            # Atenção: Mudei o nome da variável para roi_rect para não confundir
            roi_rect = self.SafeCall(
                self.GetRoiImg,
                roi_virtual_cm,
                img_shape,
                name="GetRoiImg (bola)"
            )

            if roi_rect is None:
                return None, fallback_rect

            # 3. Recortar a imagem (Igual ao predictRobot)
            x, y, w, h = roi_rect
            # Verifica se as coordenadas estão dentro do fieldReduce
            # O get_ROI_img já deve garantir, mas o slice do numpy é seguro
            crop_img = self.fieldReduce[y:y+h, x:x+w]

            if crop_img.size == 0:
                return None, roi_rect

            return crop_img, roi_rect

    def PredictRobot(self, img_shape, team: ID_Team, robot_id: ID_Robots, timestamp):
            """
            Retorna a imagem do ROI e a tupla (x, y, w, h) global.
            """
            h_img, w_img = img_shape[:2]
            
            # 1. Recupera o objeto Robô
            try:
                Tm = self.allyTeam if team == ID_Team.TEAM_ALLY else self.enemyTeam
                bot = Tm[robot_id]
            except IndexError:
                # Fallback: centro da imagem
                fallback_rect = (0, 0, int(w_img*0.5), int(h_img*0.5))
                return None, fallback_rect

            # 2. Pega ROI Virtual (cm) do Kalman
            roi_virtual_cm = self.SafeCall(
                bot.get_roi,
                image_shape=img_shape,
                t_now=timestamp,
                name=f"robot[{robot_id}].get_roi"
            )

            if roi_virtual_cm is None:
                fallback_rect = (0, 0, int(w_img*0.5), int(h_img*0.5))
                return None, fallback_rect

            # 3. Converte para Pixels e ajusta limites
            roi_rect = self.SafeCall(
                self.GetRoiImg,
                roi_virtual_cm,
                img_shape,
                name=f"GetRoiImg(robot[{robot_id}])"
            )

            if roi_rect is None:
                fallback_rect = (0, 0, int(w_img*0.5), int(h_img*0.5))
                return None, fallback_rect

            # 4. Extrai a imagem (Recorte)
            x, y, w, h = roi_rect
            roi_img = self.fieldReduce[y:y+h, x:x+w] # Assumindo que usa fieldReduce ou image passada

            return roi_img, roi_rect

    def HandleRobotLoss(self, bot, team, timestamp):
            """Gerencia falha de detecção: Predição Suave ou Reset."""
            key = (bot.id, team)
            curr_missed = self.missed_frames_robots.get(key, 0) + 1
            self.missed_frames_robots[key] = curr_missed

            if curr_missed > self.MAX_MISSED_FRAMES_ROBOT:
                # Perda real: marca o Kalman para reset na próxima detecção
                self.robot_kalman_reset_flags[key] = True
                bot.detected = False

    def _ProcessROIForRobots(self, roi_img, roi_rect, timestamp, debug=False):
        """
        Processa a ROI usando a lógica de DetectPlayers (candidatos + associação)
        para detectar robôs naquela região. Retorna uma lista de dicionários com
        as detecções (mesmo formato de DetectBotInRoi) para que o FilteredDetection
        possa atualizar o Kalman.
        """
        if roi_img is None or roi_img.size == 0:
            return []
        x0, y0, w0, h0 = roi_rect
        # Converter a ROI para HSV
        roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        # Chamar DetectPlayerCandidates com offset (coordenadas globais)
        ally_candidates, enemy_candidates = self.DetectPlayerCandidates(
            roi_img, roi_hsv, timestamp, x_offset=x0, y_offset=y0, debug=debug
        )
        # Se não houver candidatos, retorna vazio
        if not ally_candidates and not enemy_candidates:
            return []
        
        # Agora precisamos identificar cada candidato e montar um dicionário similar ao do DetectBotInRoi
        # Vamos usar a árvore de cores para identificar o robô e construir o resultado
        results = []
        all_candidates = ally_candidates + enemy_candidates
        for cand in all_candidates:
            # Identifica pela cor
            main_color = self.allyColor if cand in ally_candidates else self.enemyColor
            match, Color_p, Color_s = self.IdentifyCandidateByColor(cand, roi_hsv, main_color)
            if match is None:
                continue
            bot_id = match['robot_id']
            team = match['team']
            bot = self.GetBotById(team, bot_id)
            if bot is None:
                continue
            # Construir dicionário de resultado
            result_data = {
                "id": bot_id,
                "team": team,
                "x": cand["xcm"],
                "y": cand["ycm"],
                "theta": np.arctan2(cand["direction"][1], cand["direction"][0]),
                "direction": cand["direction"],
                "img_x": int(cand["xi"]),
                "img_y": int(cand["yi"]),
                "img_r": int(cand["ri"]),
                "bot_win": cand["windowActual"],
                "contour_global": cand.get("contour", None)
            }
            results.append(result_data)
        return results

    # ==========================================================================================
    # BLOCO 9: PIPELINE PRINCIPAL (PROC, PROCESSIMG E FILTEREDDETECTION)
    # ==========================================================================================
    def Proc(self, img, currentTime, debug: bool, isT: bool = False, force_field_detect: bool = True):
        """
        Executa a detecção do campo (sob demanda) e robôs.
        Arg:
            force_field_detect: Se True, força a execução pesada do detect_field.
                                Se False, tenta reutilizar o ROI anterior (cooVetor).
        """
        # ===========================
        # RESET ESTADO E TEMPOS
        # ===========================
        self.ResetExecutionState()
        
        if not hasattr(self, 'lastMajorTime') or self.lastMajorTime == 0:
            self.lastMajorTime = self.timer.getElapsedTime()
        self._firstTimeExec = (currentTime - self.lastMajorTime) / 1000.0
        self.debug = debug

        if img is None:
            return img

        # =========================================================
        # 1. DETECÇÃO DE CAMPO (OBRIGATÓRIA A CADA FRAME)
        # =========================================================

        # Decide se roda a detecção pesada ou usa o cache
        # Só usamos o cache se: NÃO forçado E o campo já foi detectado antes E temos o vetor salvo
        use_cache = (not force_field_detect) and self.fieldDetectedFlag and (self.viewCapture.cooVetor is not None)

        wbCmField = 0 

        if use_cache:
            try:
                # OTIMIZAÇÃO: Recorta a imagem baseada no último ROI válido
                # O cooVetor geralmente é [x, y, w, h] ou [j, i, w, h]
                x, y, w, h = self.viewCapture.cooVetor
                
                # Validação de limites para evitar crash do numpy
                if x < 0 or y < 0 or (x+w) > img.shape[1] or (y+h) > img.shape[0]:
                    raise ValueError("ROI fora dos limites da imagem")

                # Gera o fieldReduce manualmente (Processamento < 0.1ms)
                self.fieldReduce = img[y : y + h, x : x + w]
                wbCmField = w # Assume a largura do recorte
                
            except Exception as e:
                if debug: print(f"[VisSys][Proc][ERROR] - Falha ao usar cache do campo: {e}. Forçando detecção.")
                use_cache = False # Falha no cache, força detecção abaixo

        # Se não pode usar cache (ou falhou), roda a pesada detect_field (~10ms)
        if not use_cache:
            self.bmk.tic()
            wbCmField = self.DetectField(img, debug)
            self.bmk.toc("Campo")

        # Validação simples do campo (Crítico para garantir que o recorte ou detecção funcionou)
        campo_valido = (
            wbCmField != -1
            and self.fieldReduce is not None
            and self.fieldReduce.shape[0] > 10 
            and self.fieldReduce.shape[1] > 10
        )
        self.fieldDetectedFlag = campo_valido

        if not campo_valido:
            try: self.lastMajorTime = self.timer.getElapsedTime()
            except: self.lastMajorTime = currentTime
            self.frameResult = img.copy() if img is not None else None
            return self.frameResult
            
        # =========================================================
        # 2. OTIMIZAÇÃO CRÍTICA: CACHE DE HSV
        # =========================================================
        # Inicializa frameResult ANTES das detecções para que os desenhos
        # dentro de DetectBall e DetectPlayers tenham onde desenhar.
        self.frameResult = self.fieldReduce.copy() if self.fieldReduce is not None else None
        
        self.hsv_fieldReduce = cv2.cvtColor(self.fieldReduce, cv2.COLOR_BGR2HSV)

        # =========================================================
        # 3. DETECÇÃO DE OBJETOS (Usando o Cache)
        # =========================================================
        self.bmk.tic()
        self.SafeCall(self.DetectBall, self.fieldReduce, currentTime, debug, 
                        name="BALL", hsv_img=self.hsv_fieldReduce)
        self.bmk.toc("Bola")

        self.bmk.tic()
        self.SafeCall(self.DetectPlayers, self.fieldReduce, currentTime, debug, isT=isT, 
                        name="PLAYERS", hsv_img=self.hsv_fieldReduce)
        self.bmk.toc("Players")

        # ===========================
        # 4) RENDERIZAÇÃO / VISUALIZAÇÃO (APENAS DEBUG)
        # ===========================
        # Desenha os pontos/linhas do campo na imagem virtual (não afeta frameResult)
        if debug:
            self.bmk.tic()
            self.DrawFieldDebug()
            self.bmk.toc("Draw Field Debug")

        # ===========================
        # ATUALIZA TEMPO FINAL
        # ===========================
        try:
            self.lastMajorTime = self.timer.getElapsedTime()
        except Exception:
            self.lastMajorTime = currentTime

        # Garante que a imagem final sempre exista para a UI e para o fluxo de vídeo.
        if self.frameResult is None:
            if self.fieldReduce is not None:
                self.frameResult = self.fieldReduce.copy()
            elif img is not None:
                self.frameResult = img.copy()
            else:
                self.frameResult = None

        return self.frameResult

    def ProcessImg(self, img, debug: bool):
        """
        Orquestrador: Gerencia a troca entre Detecção (Proc) e Rastreamento (Filtered).
        """
        self.debug = debug
        self.frameOrigin = img
        if img is None: return img
        
        if self.timer is None: self.timer = HighPrecisionTimer(self)
        self.currentTime = self.timer.getElapsedTime()
        
        # Variável para armazenar o resultado final e evitar returns antecipados
        result = img 
        
        # =========================================================
        # MODO IMAGEM (PROCESSAMENTO ÚNICO)
        # =========================================================
        if self.emulatorMode == MODE_IMAGE:
            self._count = 0
            self.lastMajorTime = 0
            result = self.Proc(img, self.currentTime, debug, force_field_detect=True)
            
        # =========================================================
        # MODO VÍDEO (PROCESSAMENTO CONTÍNUO)
        # =========================================================
        else:
            # 1) Campo ainda NÃO detectado → Detecta aqui e avisa o proc para NÃO detectar de novo
            if not self.fieldDetectedFlag:
                wb = self.DetectField(img, debug)
                
                campo_valido = (wb != -1 and self.fieldReduce is not None)
                self.fieldDetectedFlag = campo_valido
                self._count = 0
                self.lastMajorTime = self.currentTime

                if campo_valido:
                    # OTIMIZAÇÃO: Passamos False porque ACABAMOS de detectar acima
                    result = self.Proc(img, self.currentTime, debug, force_field_detect=False)
                else:
                    result = img

            # 2) Verifica tempo para recalibração periódica
            elif (self.currentTime - self.lastMajorTime) >= self.newProcTime:
                wb = self.DetectField(img, debug)
                campo_valido = (wb != -1 and self.fieldReduce is not None)
                self.fieldDetectedFlag = campo_valido
                self.lastMajorTime = self.currentTime
                self._count = 0

                if campo_valido:
                    # OTIMIZAÇÃO: Passamos False, pois detect_field já rodou acima
                    result = self.Proc(img, self.currentTime, debug, force_field_detect=False)
                else:
                    result = img
            
            # 3) Warm-up do Kalman (frames iniciais)
            elif self._count < 12000: # WARMUP_FRAMES
                self._count += 1
                result = self.Proc(img, self.currentTime, debug, force_field_detect=False)
            
            # 4) Rastreamento rápido (Filtered Detection)
            else:
                try:
                    self.FilteredDetection(img, self.currentTime, debug)
                    # Assume-se que FilteredDetection atualiza self.frameResult internamente
                    result = getattr(self, "frameResult", img) 
                except Exception as e:
                    if debug: print(f"[VisSys][ProcessImg][ERROR] - Erro no Tracking: {e}. Reiniciando detecção.")
                    
                    # 1. Marca que perdemos a garantia de onde está o campo
                    self.fieldDetectedFlag = False
                    self._count = 0

                    # 2. Chama a Proc() forçando a redetecção e SALVA o resultado
                    result = self.Proc(img, self.currentTime, debug, force_field_detect=True)

        # =========================================================
        # CÁLCULO DE dT PARA FÍSICA/PREDIÇÃO (Agora sempre executa!)
        # =========================================================
        tmf = self.timer.getElapsedTime()
        self.dT = tmf - self.currentTime

        # Atualiza a variável de classe por segurança e retorna o frame processado
        if result is None:
            result = self.frameResult if self.frameResult is not None else (img.copy() if img is not None else None)
        if result is None:
            result = img.copy() if img is not None else None
        if isinstance(result, np.ndarray) and result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # =========================================================
        # RENDERIZAÇÃO EXTRAS (APENAS DEBUG)
        # =========================================================
        # Observação: os desenhos básicos (círculos e nomes de robôs/bola) já foram
        # feitos dentro de DetectBall e DetectPlayers (sempre ativos).
        # Portanto, removemos DrawAllRobots e DrawBallDebug para evitar duplicação.
        if self.debug:
            self.DrawKalmanDebug(self.currentTime)
            
        return self.frameResult

    def FilteredDetection(self, img, currentTime, debug=False):
        """
        Pipeline otimizado:
        1. Validações e Recorte do Campo.
        2. Predição e Atualização da Bola.
        3. Predição e Atualização dos Robôs (Delegando lógica interna para a classe Robot).
        """
        print("[VisSys][FilteredDetection][INFO] - Iniciando rastreamento rápido")
        # ==========================================================
        # 0) Validações e Sanity Checks
        # ==========================================================
        if img is None or not self.fieldDetectedFlag:
            return self.Proc(img, currentTime, debug)
        
        if not hasattr(self.viewCapture, "cooVetor") or self.viewCapture.cooVetor is None:
            return self.Proc(img, currentTime, debug)

        # Extrai coordenadas e valida tamanho mínimo (Sanity Check)
        try:
            x_w, y_w, w_w, h_w = map(int, self.viewCapture.cooVetor)
        except ValueError:
            return self.Proc(img, currentTime, debug)

        if self.prop_px_cm > 0:
            w_r = w_w / self.prop_px_cm
            h_r = h_w / self.prop_px_cm
            if w_r < 100 or h_r < 100: 
                if debug: print("[VisSys][FilteredDetection][DEBUG] - Campo muito pequeno. Resetando.")
                return self.Proc(img, currentTime, debug)

        # ==========================================================
        # 1) Extração do FieldReduce
        # ==========================================================
        H0, W0 = img.shape[:2]
        x_w = max(0, min(x_w, W0 - 1))
        y_w = max(0, min(y_w, H0 - 1))
        w_w = max(1, min(w_w, W0 - x_w))
        h_w = max(1, min(h_w, H0 - y_w))

        self.fieldReduce = img[y_w:y_w + h_w, x_w:x_w + w_w].copy()
        H, W = self.fieldReduce.shape[:2]

        self.frameResult = self.fieldReduce.copy()
        
        if hasattr(self, "virtual") and self.virtual is not None:
            self.virtualImg = self.virtual.copy()

        self.imgHSV = cv2.cvtColor(self.fieldReduce, cv2.COLOR_BGR2HSV)

        # ==========================================================
        # LIMPEZA DAS MÁSCARAS DE DEBUG (evita persistência de desenhos)
        # ==========================================================
        self.binaryBall = np.zeros((H, W), dtype=np.uint8)
        self.binaryPlayers = np.zeros((H, W), dtype=np.uint8)
        self.binaryAllies = np.zeros((H, W), dtype=np.uint8)
        self.binaryAllyConfirmed = np.zeros((H,W),dtype=np.uint8)
        self.binaryObjects = np.zeros((H, W), dtype=np.uint8)
        self.binReduceField = np.zeros((H, W), dtype=np.uint8)
        self.binField = np.zeros((H, W), dtype=np.uint8)
        # ==========================================================

        # ==========================================================
        # 2) DETECÇÃO DA BOLA
        # ==========================================================
        roi_ball_img, roi_ball_rect = self.PredictBall((H, W), currentTime)
        
        ball_found = False
        self.bmk.tic()
        if roi_ball_img is not None:
            ball_data = self.SafeCall(
                self.SearchBall, roi_img=roi_ball_img, roi_rect=roi_ball_rect,
                timestamp=currentTime, debug=debug, name="SearchBall"
            ) or {}
            ball_found = ball_data.get("found", False)
        self.bmk.toc("Bola")
        
        if ball_found:
            xb, yb = int(ball_data["img_x"]), int(ball_data["img_y"])
            r_px = int(ball_data.get("img_r", 4))
            xcm, ycm = float(ball_data["x"]), float(ball_data["y"])

            # Watchdog: se a bola ficou perdida tempo demais, o Kalman foi marcado
            # para reset. Reinicializa antes de reatualizar para evitar um salto de
            # inovação a partir de um estado obsoleto.
            if getattr(self, "ball_kalman_reset_flag", False) or not self.ball.kalman_initialized:
                self.ball.reset()
                self.ball_kalman_reset_flag = False
                # Primeira medição após reset: inicializa o filtro.
                self.ball.setPosition(xcm, ycm, self.ballRadiusP, currentTime)
            else:
                # updatePosition deriva direção/theta do movimento e alimenta o Kalman
                # com o theta derivado (setPosition zeraria direção e theta a cada frame).
                self.ball.updatePosition(xcm, ycm, self.ballRadiusP, currentTime)

            self.ball.setImgPosition(xb, yb, r_px)
            self.ball.detected = True
            self.missed_frames_ball = 0

            if self.binaryBall is not None:
                cv2.circle(self.binaryBall, (xb, yb), int(r_px * 1.3), 255, -1)
            if debug:
                cv2.circle(self.frameResult, (xb, yb), r_px+1, (0,0,255), 2)
        else:
            self.missed_frames_ball = getattr(self, "missed_frames_ball", 0) + 1
            if self.missed_frames_ball > self.MAX_MISSED_FRAMES_BALL:
                self.ball.detected = False
                self.ball_kalman_reset_flag = True

        # ==========================================================
        # 3) DETECÇÃO DOS ROBÔS
        # ==========================================================
        self.bmk.tic()
        robots_found = self.SafeCall(
            self.SearchBots,
            img=self.fieldReduce, # SearchBots recorta internamente via predictRobot
            timestamp=currentTime,
            debug=debug
        ) or []
        self.bmk.toc("Players")

        updated_keys = set()

        for r in robots_found:
            try:
                rid = r["id"]
                team = r["team"]
                key = (rid, team)
                
                if key in updated_keys: continue
                updated_keys.add(key)

                bot = self.GetBotById(team, rid)
                if bot is None: continue

                # Extração dos dados
                mx, my = float(r["x"]), float(r["y"])
                direction = r["direction"] # Vetor np.array([dx, -dy])
                theta = float(r["theta"])  # Apenas para debug visual, pois updatePosition recalcula
                
                # Dados visuais
                x_img, y_img = int(r["img_x"]), int(r["img_y"])
                r_img = int(r.get("img_r", 8))
                
                # A IMAGEM DO ROBÔ (WINDOW)
                # search_bots deve retornar isso na chave "bot_win"
                bot_window = r.get("bot_win", None) 

                # --- Lógica de Reset (Watchdog) ---
                missed = int(self.missed_frames_robots.get(key, 0))
                
                # Se perdeu por muito tempo, resetamos o robô.
                # Isso fará o próximo updatePosition reinicializar o Kalman.
                if missed > self.MAX_MISSED_FRAMES_ROBOT or self.robot_kalman_reset_flags.get(key, False):
                    bot.reset() 
                    self.robot_kalman_reset_flags[key] = False

                # --- Gating de distância (fail-safe, mesmo princípio do DetectPlayers) ---
                # Se o Kalman já está de pé, rejeita um salto implausível (ex: candidato de
                # outro robô que "vazou" para dentro da ROI numa colisão) antes de alimentar
                # o filtro com uma medição ruim.
                if bot.kalman_initialized:
                    last_pos = getattr(bot, "position_filtered", None)
                    if last_pos is not None:
                        jump = float(np.hypot(mx - last_pos[0], my - last_pos[1]))
                        if jump > self.MAX_JUMP_CM:
                            if debug:
                                print(f"[VisSys][FilteredDetection][DEBUG] - Robô {key}: salto de {jump:.1f}cm rejeitado.")
                            # Trata este frame como "não detectado" para esse robô, para que
                            # o watchdog final (fallback de predição / contagem de perdido)
                            # seja acionado normalmente.
                            updated_keys.discard(key)
                            continue

                # --- ATUALIZAÇÃO CENTRALIZADA NO ROBÔ ---
                # Mesma regra já usada acima para a bola e em DetectPlayers: sem o Kalman
                # inicializado (primeira medição ou logo após um reset), setPosition precisa
                # rodar primeiro; só depois disso updatePosition (predict+correct) é seguro.
                if not bot.kalman_initialized:
                    bot.setPosition(mx, my, direction, bot_window, currentTime)
                else:
                    bot.updatePosition(mx, my, direction, bot_window, currentTime)
                
                # Atualização extra de propriedades visuais (Bounding Box global)
                bot.updtPositionImg(x_img, y_img, r_img)
                bot.detected = True
                
                self.missed_frames_robots[key] = 0

                # Desenho de Debug
                if debug:
                    color = (255, 0, 0) if team == ID_Team.TEAM_ALLY else (0, 0, 255)
                    end_pt = (int(x_img + 15*np.cos(theta)), int(y_img - 15*np.sin(theta)))
                    cv2.arrowedLine(self.frameResult, (x_img, y_img), end_pt, color, 1)
                    cv2.putText(self.frameResult, str(rid), (x_img-5, y_img-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            except Exception as e:
                if debug:
                    print(f"[VisSys][FilteredDetection][ERROR] - Erro ao processar robô {key}: {e}")
                    traceback.print_exc()

        # ==========================================================
        # 4) Watchdog (Robôs perdidos)
        # ==========================================================
        # Primeiro, garantir que robôs detectados por outros métodos (ex: ROI) sejam considerados
        for bot, team in [(b, ID_Team.TEAM_ALLY) for b in self.allyTeam] + \
                        [(b, ID_Team.TEAM_ENEMY) for b in self.enemyTeam]:
            key = (bot.id, team)
            if bot.detected and key not in updated_keys:
                updated_keys.add(key)
                self.missed_frames_robots[key] = 0

        # Agora, aplicar watchdog para os que não foram detectados
        for bot, team in [(b, ID_Team.TEAM_ALLY) for b in self.allyTeam] + \
                        [(b, ID_Team.TEAM_ENEMY) for b in self.enemyTeam]:
            key = (bot.id, team)
            if key not in updated_keys:
                self.SafeCall(self.HandleRobotLoss, bot, team, currentTime)

        return self.frameResult


# ==========================================================================================
# TESTE DA CLASSE
# ==========================================================================================
if __name__ =='__main__':
    print("[VisSys][main][INFO] - Utilizada em função de main")