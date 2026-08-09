# ==========================================================================================
# MÓDULO DE DETECÇÃO VSS (Vision System Soccer) -- v3.3.22 (BETA)
# ==========================================================================================
import cv2
import numpy as np
from timer import *

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

    def __init__(self,config: EConfig = EConfig(), debug:bool = False, capture: Capture = None, UseCuda:bool = False, GPUType:GPUType = None):
        self.CreateObjs()
        self._countProcess = 0
        self.bmk = Benchmark()

        # Watchdog: nº de frames perdidos tolerados antes de resetar o Kalman
        self.MAX_MISSED_FRAMES_BALL = 120 #2 segundos a 60 FPS e 4 segundos a 30FPS
        self.MAX_MISSED_FRAMES_ROBOT = 120
        self.missed_frames_ball = 0
        self.ball_kalman_reset_flag = False
        self.missed_frames_robots = {}
        self.robot_kalman_reset_flags = {}

        # Gating de distância (fail-safe): rejeita saltos implausíveis antes do Kalman
        self.MAX_JUMP_CM = 60.0

        self._count: int            = 0
        self._firstTimeExec: int    = 0
        self.config:EConfig    = config
        self._capture:Capture  = capture
        self._hasCuda           = UseCuda 
        self._GPUType           = GPUType

        if self._capture is not None:
            self._capture.GPUMode(self._hasCuda)

        #Verifica 
        self.GPUimg         = None
        self.CPUimg         = None 
        self.emulatorMode   = MODE_IMAGE                #supõe que é imagem

        #verifica o timer necessário para realizar as previsões
        self.timer: HighPrecisionTimer    = None 

        #tempos necessários
        self.dT = 0 
        ''' Aqui é o intervalo de tempo que leva para processar'''


        #gera o objeto para utilizar o cuda
        if(self._hasCuda):
            #variável para guardar o endereço da imagem principal
            self.GPUimg = cv2.cuda.GpuMat()


        #configurações do campo comprimento e largura
        self.fieldWidth             =  150                    # largura do campo
        self.fieldHeight            = 130                     # altura do campo
        self.prop_px_cm             = 1                     # proporção pixel para cm
        self.prop_px_cm_virtual     = 3                     # proporção pixel para cm na imagem virtual
        self.min_diag = (7.5 / 4) * np.sqrt(2) * self.prop_px_cm
        
        self.homography_matrix      = None                  # matrix de homografia entre imagem real e virtual
        self.inv_homography_matrix  = None                  # matrix inversa de homografia, relação entre a imagem virtual e a imagem real (caso necessário)

        #variável de debug
        self.debug:bool   = debug                                  # verifica se o processamento usará ou não o debug

        # Coordenada do ponto de origem do novo sistema de coordenadas
        self.xnv     = 67                     
        self.ynv     = 402        

        self.coordOrigin = np.array([self.xnv,self.ynv])

        #variáveis de controle de tempo de execução
        self.lastMajorTime = 0 
        self.currentTime   = 0                  # tempo atual de execução

        self.newSendTime = 0.02

        #Tamanho padrão da bola
        self.ballRadiusP = 2.135 #cm

        #tamanho do campo para utilizar
        self.modDpCm     = 0        
        self.fieldDetectedFlag = False 

        #coordenadas dos pontos importantes na imagem virtual
        #Essas coordenadas são em pixels, para passar para o sistema de coordenadas O'
        #Necessário utilizar a função getVirtualPoint()

        #extremos do campo virtual
        self.fieldP1v  =   np.array([97,12])
        self.fieldP2v  =   np.array([547,12])
        self.fieldP3v  =   np.array([547,402])
        self.fieldP4v  =   np.array([97,402])
        
        #centro do campo virtual
        self.fieldCenterv  =   np.array([322,207])

        #pivots virtual
        self.PA1v   =   np.array([210,87])
        self.PA2v   =   np.array([210,207])
        self.PA3v   =   np.array([210,327])
        self.PE1v   =   np.array([435,87])
        self.PE2v   =   np.array([435,207])
        self.PE3v   =   np.array([435,327])

        #area goleiro aliado virtual
        self.GA1v   =   np.array([97,102])
        self.GA2v   =   np.array([142,102])    
        self.GA3v   =   np.array([142,312])  
        self.GA4v   =   np.array([97,312])

        #area interna do goleiro aliado virtual
        self.GAI1v  =   np.array([67,147])
        self.GAI2v  =   np.array([97,147])
        self.GAI3v  =   np.array([97,267])
        self.GAI4v  =   np.array([67,267])
        
        #area goleiro aliado virtual
        self.GE1v   =   np.array([502,102])
        self.GE2v   =   np.array([547,102])    
        self.GE3v   =   np.array([547,312])  
        self.GE4v   =   np.array([502,312])

        #area interna do goleiro aliado virtual
        self.GEI1v  =   np.array([547,147])
        self.GEI2v  =   np.array([577,147])
        self.GEI3v  =   np.array([577,267])
        self.GEI4v  =   np.array([547,267])

        # área aliada
        #meios dos lados virtual
        self.fieldP12v  =   np.array([322,12])
        self.fieldP34v  =   np.array([322,402])



        # Configurações gerais
        self.offSetWindow       = 10                 # margem extra de janela
        self.offSetErode        = 0                  # nº padrão de erosões
        self.dimMatrix          = 25                 # dimensão da matriz de convolução
        self.Thrashhold         = 235                # limiar de binarização
        self.pixelWidth         = 1

        # Imagens
        self.frameOrigin        = None
        self.ballImg            = None
        self.fieldReduce        = None
        self.frameResult        = None
        self.imgReduce          = None
        self.binaryAllies       = None
        self.binaryAllyConfirmed = None

        self.virtual            = cv2.imread("src/data/images/CampoVirtual.png")
        self.virtualImg         = self.virtual.copy()   # cópia manipulável

        #Imagens binarizadas utilizadas no código
        self.binaryObjects      = np.zeros((1, 1), dtype=np.uint8)              # Imagem binária dos objetos
        self.binaryPlayers      = np.zeros((1, 1), dtype=np.uint8)              # Imagem Binária dos Jogadores
        self.binaryBall         = np.zeros((1, 1), dtype=np.uint8)              # Imagem binária da bola
        self.binReduceField     = np.zeros((1, 1), dtype=np.uint8)              # Imagem binarizada do campo reduzido tratada
        self.binField           = np.zeros((1, 1), dtype=np.uint8)              # Imagem binarizada do campo original tratada


        #Cores dos jogadores salvas para salvar nos jogadores
        self.ballColor          = None              # Cor da bola
        self.allyColor          = None              # Cor do time aliado
        self.enemyColor         = None              # Cor do time inimigo
        self.goalAllyColor1     = None              # cor 1 do goleiro aliado
        self.goalAllyColor2     = None              # cor 2 do goleiro aliado
        self.atk1AllyColor1     = None              # cor 1 do atacante 1
        self.atk1AllyColor2     = None              # cor 2 do atacante 1
        self.atk2AllyColor1     = None              # cor 1 do atacante 2
        self.atk2AllyColor2     = None              # cor 2 do atacante 2

        #cores padrões dos objetos para o sistema:    #Carrega os vetores de cores claras e escuras de objetos gerais 
        self.objectsDarkColor   = np.array([0,10,130]) #[0,10,150]
        self.objectsLightColor  = np.array([179,255,255])

        #definições estruturais do código
        self.playerRadius       = 0
        self.mainColorRadius    = 0
        self.secColorRadius     = 0

        #definições úteis dentro do código
        self.ally_lower_bound   = None          # valor mínimo para detectar aliados
        self.ally_upper_bound   = None          # valor máximo para detectar os aliados
        self.enemy_lower_bound  = None          # valor mínimo para detectar inimigos
        self.enemy_upper_bound  = None          # valor máximo para detectar inimigos

        # variáveis internas para realizar o tratamento de dados
        self.playersCount       = 0
        self.alliesCount        = 0
        self.enemiesCount       = 0
        self.fieldDetectionFailCount  = 0
        self.maxFieldFailures = 10

        self.playersWindows     = [None, None, None, None, None, None]

        self.alliesWindows      = [None, None, None]
        self.enimiesWindows     = [None, None, None]

        #lista de threads a serem utilizadas pelo objeto
        self._threads           = []

        #Variável importante para ditar quanto tempo até a próxima atualização de dados
        self.newProcTime        = 10000
        
        # Variável da identificação de cores
        self.colorTree = TreeColors()

        #Extrai os dados do objeto de configuração 
        self.ToMineData()


        # ===========================================================================
        # CACHE
        # [OTIMIZAÇÃO] Cache de estruturas para detect_field
        # Evita recriar matrizes a cada frame
        self.kernel_blur = (5, 5) 
        self.kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
        
        self.ptsSource_buffer = np.zeros((4, 2), dtype=np.float32)
        # Cache para evitar alocação de numpy arrays no loop
        self.ptsSource_cache = np.zeros((4, 2), dtype=np.float32)
        self.ptsFinal_cache = np.array([
                self.fieldP1v, self.fieldP2v,
                self.fieldP3v, self.fieldP4v
            ], dtype=np.float32)
    
    
        # Objetos utilizados para estrutura do código.
        self.struct_ellipse5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.struct_rect11   = cv2.getStructuringElement(cv2.MORPH_RECT,   (11,11))

        #construir o campo
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
        print("[VisionSystem]: Configurações carregadas")
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
    
    def GetViewCapture(self):
        '''Retorna a janela de interesse (ROI) do sistema de visão.'''
        return self.viewCapture

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
                if self.debug: print(f"[VS] Erro Bola Protobuff: {e}")

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
                        if self.debug: print(f"[VS] Erro Robot Ally ID {robot.id}: {e}")

        # Time Inimigo -> Blue
        if self.enemyTeam:
            for robot in self.enemyTeam:
                if robot is not None and robot.detected:
                    try:
                        robot_pb = frame.robots_blue.add()
                        fill_robot_proto(robot, robot_pb)
                    except Exception as e:
                        if self.debug: print(f"[VS] Erro Robot Enemy ID {robot.id}: {e}")

        return frame

    def GetBotById(self, team:ID_Team, bot_id:ID_Robots) -> Robot:
        '''Retorna o robô pelo time + ID.'''
        if team == ID_Team.TEAM_ALLY:
            return self.allyTeam[bot_id]
        else:
            return self.enemyTeam[bot_id]

    def MatchesRobot(self, bot: Robot, primary, secondary) -> bool:
        '''Verifica se a combinação de cores detectada corresponde ao robô esperado.'''
        team = bot.team
        robot_id = bot.id
        main_color = bot.colorTeam

        match = self.colorTree.find_by_colors(main_color, primary, secondary)
        if match is None:
            return False

        return match['robot_id'] == robot_id and match['team'] == team

    #Puxando as imagens de debug
    def GetDebugImages(self):
        return self.binaryObjects, self.binaryBall, self.binaryPlayers, self.binaryAllies 
        
    def GetShape(self, img):
        """Retorna (altura, largura) independente se é CPU (numpy) ou GPU (UMat)."""
        if isinstance(img, cv2.UMat):
            return img.get().shape[:2]
        return img.shape[:2]

    def CheckFieldReset(self):
        '''Se houve falhas consecutivas demais na detecção do campo, reseta robôs, bola e homografia.'''
        if self.fieldDetectionFailCount >= self.maxFieldFailures:
            print(f"[VisionSystem] RESET: {self.fieldDetectionFailCount} falhas consecutivas na detecção do campo")
            for bot in (*self.allyTeam, *self.enemyTeam):
                bot.reset()
            if hasattr(self, "ball"):
                self.ball.reset()

            self.homography_matrix = None
            self.inv_homography_matrix = None
            self.fieldDetectedFlag = False
            self.fieldDetectionFailCount = 0

            if self.debug:
                print("[FIELD_RESET] Sistema resetado devido a falhas persistentes na detecção do campo")

    # ==========================================================================================
    # BLOCO 2: PROCESSAMENTO DE IMAGEM, CORES, DESENHO E UTILITÁRIOS
    # ==========================================================================================
    def LoadImage(self, imgPath):
        '''Carrega imagem do disco (CPU).'''
        self.imgOrigim = cv2.imread(imgPath)

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
            kernel = self.kernel_morph
            binImgProc = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel, iterations=it)
            binImgProc = cv2.morphologyEx(binImgProc, cv2.MORPH_CLOSE, kernel, iterations=max(1, it - 1))
            return binImgProc
        except Exception as e:
            print(f"[SystemVision][TRAIT_NOISE]: Erro ao tratar ruído — {e}")
            return binImg

    def GetObject(self,img):
        '''Retorna o maior contorno da imagem, ou None.'''
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def GetPerspective(self, contour):
        '''Retorna (x1, y1, x2, y2) do retângulo delimitador do contorno, ou None se inválido.'''
        if contour is None or len(contour) == 0:
            return None
        contour = np.array(contour).astype(np.int32)
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, x + w, y + h
    
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
            print(f"[SystemVision][HIGHLIGHT_IMG]: Erro ao realçar imagem — {e}")
            return img

    def ReduceWindow(self, img, coorVetor, d=10):
        '''Recorta `img` na subjanela [x,y,w,h] (+ margem `d`) via transformação de perspectiva.'''
        try:
            x, y, w, h = map(int, coorVetor)

            if w <= 0 or h <= 0:
                print("[SystemVision][REDUCE_WINDOW]: Dimensões inválidas de recorte.")
                return img

            # Limita margem para evitar índices negativos
            d = max(0, min(d, x, y))

            # Define pontos da área de interesse (com margem)
            firstPoints = np.float32([
                [x - d, y - d],
                [x + w + d, y - d],
                [x - d, y + h + d],
                [x + w + d, y + h + d]
            ])

            # Pontos destino (retângulo final)
            lastPoints = np.float32([
                [0, 0],
                [w, 0],
                [0, h],
                [w, h]
            ])

            # Calcula matriz de transformação perspectiva
            matrizPerspectiva = cv2.getPerspectiveTransform(firstPoints, lastPoints)

            # Garante que dimensões sejam inteiras e válidas
            w, h = int(max(1, w)), int(max(1, h))

            # Aplica a transformação de perspectiva
            img_Reduce = cv2.warpPerspective(img, matrizPerspectiva, (w, h))

            return img_Reduce

        except Exception as e:
            print(f"[SystemVision][REDUCE_WINDOW]: Erro ao reduzir janela — {e}")
            return img
   
    def ReduceField(self, BinImg, Img, fieldWidth, d=10):
        '''Recorta BinImg/Img para o bounding box do maior contorno (o campo), com margem `d`.'''
        contours, _ = cv2.findContours(BinImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[SystemVision]/[REDUCE_FIELD]: Nenhum contorno encontrado.")
            return BinImg, Img, [0, 0, 0, 0]

        try:
            # Selecionar contornos úteis (descartar ruidos pequenos)
            contours = [c for c in contours if cv2.contourArea(c) > 200]  
            if not contours:
                print("[REDUCE_FIELD]: Contornos muito pequenos.")
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
            print(f"[SystemVision][REDUCE_FIELD]: Erro ao reduzir o campo: {e}")
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
            print(teamList[i].team, teamList[i].id)
            print("Posição: x =", teamList[i].pos[0], " y =", teamList[i].pos[1])
        
        print("====================")

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
    
    def UpSaturation(self, img, boost: int = 50):
        '''Aumenta a saturação (canal S) de uma imagem BGR em `boost`.'''
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.int16) + boost, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def FindBinaryContours(self, image, lower_bound, upper_bound):
        '''Encontra contornos filtrando a imagem por faixa HSV.'''
        lower = np.array(lower_bound)
        upper = np.array(upper_bound)

        if image is None:
            print("[VisionSystem]: Em FindBinaryContours() a Imagem é None")
            return [], None

        if image.shape[1] < 30:
            print("[VisionSystem]: Em FindBinaryContours() a janela é muito pequena, provável que nem exista")
            return [], None
        
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # mask_in_range trata o wrap circular do Hue (bounds vêm de create_color_bounds)
        binaryImage = self.MaskInRange(imageHSV, lower, upper)

        structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, structuringElement)
        binaryImage = cv2.erode(binaryImage, structuringElement, iterations=1)

        contours, _ = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours
    
    def SortPoints(self, points):
        '''Ordena 4 pontos como [top_right, top_left, bottom_left, bottom_right].'''
        points = sorted(points, key=lambda p: (p[1], p[0]))
        top_points = sorted(points[:2], key=lambda p: p[0])
        bottom_points = sorted(points[2:], key=lambda p: p[0])
        return np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.int32)

    def DetectSquares(self,img_bin, min_diag=20):
        '''Filtra uma imagem binarizada, mantendo só os contornos com formato de quadrado.'''
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_bin, []
        mask = np.zeros_like(img_bin)
        contours_treat = []
        for contour in contours:
            if self.IsSquare(contour, min_diag):
                cv2.drawContours(mask, [contour], -1, 255, -1)
                contours_treat.append(contour)
        bin_res = cv2.bitwise_and(img_bin, mask)
        return bin_res, contours_treat

    def SafeCall(self, func, *args, name="", **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[VS][SAFE_CALL] Erro ao executar {name}: {e}")
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
        cv2.circle(self.virtualImg, (self.xnv, self.ynv), 3, (0, 255, 255), -1)

    def DrawBallDebug(self):
        """Desenha a bola no frameResult quando o debug estiver ativo."""
        if self.frameResult is None or self.ball is None:
            return

        if hasattr(self.ball, "xb") and hasattr(self.ball, "yb") and hasattr(self.ball, "rb"):
            xb = int(getattr(self.ball, "xb", 0))
            yb = int(getattr(self.ball, "yb", 0))
            rb = int(getattr(self.ball, "rb", 4))
            cv2.circle(self.frameResult, (xb, yb), max(rb + 2, 4), (0, 0, 255), 2)
            cv2.putText(self.frameResult, "B", (xb, yb - rb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def DrawAllRobots(self):
        '''
            Desenhando todos os robôs na imagem final
        '''
        for bot in self.allyTeam:
            if bot.getStatus():
                self.DrawPlayerCircle(self.frameResult, bot)
                self.DrawPlayerVirtual(bot)

        for bot in self.enemyTeam:
            if bot.getStatus():
                self.DrawPlayerCircle(self.frameResult, bot)
                self.DrawPlayerVirtual(bot)

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
                print(f"[DrawFilteredPosition] erro: {e}")
            
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
            print("[ERROR]: Matriz de homografia singular e foi substituída por uma matriz identidade.")

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

    def IsSquare(self, contour, min_diag=20):
        """
        Verifica se o contorno corresponde aproximadamente a um quadrado.
        Retorna True se for quadrado, False caso contrário.
        """
        perimetro = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimetro, True)
        # Precisa ter 4 lados
        if len(approx) != 4:
            return False
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        diag = np.hypot(w, h)
        # Aproximadamente quadrado e com tamanho mínimo
        return 0.7 <= aspect_ratio <= 1.3 and diag >= min_diag

    def IsSquareContour(self, contorno):
        """
        Verifica se o contorno é aproximadamente quadrado
        usando medidas geométricas simples (mais rápido).
        """
        area = cv2.contourArea(contorno)
        if area < 10:  # ignora ruídos muito pequenos
            return False

        x, y, w, h = cv2.boundingRect(contorno)
        aspect = w / float(h)
        extent = area / (w * h)

        # Quadrados têm aspecto ~1 e extent próximo de 1
        return 0.7 <= aspect <= 1.3 and extent > 0.8

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
    
    def DetectFieldOnce(self, img, debug):
        '''
            Método auxiliar para detectar o campo uma vez e retornar se foi válido.
        '''
        wb = self.DetectField(img, debug)

        campo_valido = (
            wb != -1 and
            abs(wb - self.fieldWidth) < 20 and
            self.fieldReduce is not None
        )

        self.fieldDetectedFlag = campo_valido
        return campo_valido

    def DetectField(self, img, debug):
        """
        Detecta o campo com no máximo DUAS tentativas.
        Retorna o tamanho do campo detectado em centímetros ou -1 se falhar.
        """

        if img is None:
            print("[VisionSystem]: A imagem é nula!!")
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
                print("[VisionSystem]: Erro durante detecção:", e)
                traceback.print_exc()
                continue  # vai para segunda tentativa

        # FIM DAS DUAS TENTATIVAS

        if not campo_detectado:
            self.fieldDetectionFailCount += 1
            if debug:
                print(f"[FIELD_DETECTION] Falha #{self.fieldDetectionFailCount}")
        else:
            self.fieldDetectionFailCount = 0

        self.CheckFieldReset()

        self.frameResult = (self.fieldReduce.copy()
                            if self.fieldReduce is not None
                            else img.copy())

        # ---------------------------------------------------------
        
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
            print("FIELDREDUCE É NONE")
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

        if not hasattr(self, "_enemy_last_pos"):
            self._enemy_last_pos = {}   # {slot: (xcm, ycm)}
        if not hasattr(self, "_ally_last_pos"):
            self._ally_last_pos = {}    # {bot_id: (xcm, ycm)}

        self.playersCount = self.enemiesCount = self.alliesCount = 0
        for bot in (*self.enemyTeam, *self.allyTeam):
            bot.setStatus(False)

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
        ellipse5 = self.struct_ellipse5
        rect11 = self.struct_rect11

        obj_mask = cv2.inRange(imgHSV, self.objectsDarkColor, self.objectsLightColor)
        if self.ball.status:
            xb, yb = int(self.ball.xb) - x_offset, int(self.ball.yb) - y_offset
            r = 5
            h, w = obj_mask.shape[:2]
            y1b, y2b = max(0, yb - r), min(h, yb + r)
            x1b, x2b = max(0, xb - r), min(w, xb + r)
            obj_mask[y1b:y2b, x1b:x2b] = 0

        closed_mask = cv2.erode(obj_mask, ellipse5, iterations=1)
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, rect11)

        winSize = int(18 * self.prop_px_cm)
        half_win = winSize // 2
        playerRadius = (7.5 / 2) * np.sqrt(2) * self.prop_px_cm
        mainColorRadius = (7.5 / 4) * np.sqrt(5) * self.prop_px_cm
        single_area = np.pi * playerRadius ** 2   # área de UM robô
        NOISE_RADIUS_MIN = 0.2 * playerRadius

        ally_candidates = []
        enemy_candidates = []

        raw_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in raw_contours:
            if self.playersCount >= max_players:
                break
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < NOISE_RADIUS_MIN:
                continue

            if self.binaryPlayers is not None:
                cv2.drawContours(self.binaryPlayers, [cnt], -1, 255, -1, offset=(x_offset, y_offset))

            area = cv2.contourArea(cnt)
            if area <= 0:
                area = np.pi * r * r

            # ========== NOVA ESTIMATIVA DE n_est (considera sobreposição) ==========
            if area > 2.2 * single_area:
                n_est = 3
            elif area > 1.2 * single_area:
                n_est = 2
            else:
                n_est = 1
            n_est = min(n_est, max_players - self.playersCount)
            # ======================================================================

            if debug and self.frameResult is not None:
                color_circle = (0, 165, 255) if n_est > 1 else (0, 255, 0)
                cv2.circle(self.frameResult, (int(cx) + x_offset, int(cy) + y_offset), int(r) + 5, color_circle, 2)

            if n_est <= 1:
                result = self._BuildPlayerCandidate(
                    img, imgHSV, cx, cy, half_win, playerRadius, mainColorRadius,
                    x_offset, y_offset, cnt=cnt
                )
                if result is not None:
                    team_is_enemy, cand = result
                    (enemy_candidates if team_is_enemy else ally_candidates).append(cand)
                    self.playersCount += 1
            else:
                centers = self.ResolveCollisionBlob(
                    img.shape, cnt, cx, cy, r, n_est, winSize, timestamp,
                    x_offset=x_offset, y_offset=y_offset, imgHSV=imgHSV
                )
                for (cx_i, cy_i) in centers[:n_est]:
                    result = self._BuildPlayerCandidate(
                        img, imgHSV, cx_i, cy_i, half_win, playerRadius, mainColorRadius,
                        x_offset, y_offset, cnt=None
                    )
                    if result is not None:
                        team_is_enemy, cand = result
                        (enemy_candidates if team_is_enemy else ally_candidates).append(cand)
                        self.playersCount += 1
                        if self.playersCount >= max_players:
                            break

        return ally_candidates, enemy_candidates


    def _BuildPlayerCandidate(self, img, imgHSV, cx, cy, half_win, playerRadius, mainColorRadius,
                               x_offset=0, y_offset=0, cnt=None):
        """
        Helper privado da PARTE 1. Constrói um candidato a robô a partir de um
        centro estimado (cx, cy) em coordenadas LOCAIS a `img`/`imgHSV`.

        `x_offset`/`y_offset` (posição da janela no referencial global) são
        somados apenas nos campos que saem "para fora" desta janela (xi, yi,
        x_m, y_m e a conversão px->cm via TransformPoint) -- o recorte
        "windowActual" continua local, pois só é usado para achar a cor
        dentro da própria janela.
        """
        x1 = max(0, int(cx - half_win))
        y1 = max(0, int(cy - half_win))
        x2 = min(img.shape[1], int(cx + half_win))
        y2 = min(img.shape[0], int(cy + half_win))
        windowActual = img[y1:y2, x1:x2]
        if windowActual.size == 0:
            return None
        hsv = imgHSV[y1:y2, x1:x2]
        mask_ally = self.MaskInRange(hsv, self.ally_lower_bound, self.ally_upper_bound)
        mask_enemy = self.MaskInRange(hsv, self.enemy_lower_bound, self.enemy_upper_bound)
        ally_area = cv2.countNonZero(mask_ally)
        enemy_area = cv2.countNonZero(mask_enemy)
        total_area = max(ally_area + enemy_area, 1)
        ally_ratio = ally_area / total_area
        enemy_ratio = enemy_area / total_area

        if ally_ratio > 0.4:
            team_is_enemy = False
        elif enemy_ratio > 0.4:
            team_is_enemy = True
        else:
            return None

        curr_mask = mask_enemy if team_is_enemy else mask_ally
        contour = max(cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                    key=cv2.contourArea, default=None)
        if contour is None:
            return None
        (x_m, y_m), rc = cv2.minEnclosingCircle(contour)
        x_m += x1
        y_m += y1
        min_rc = (0.75 if team_is_enemy else 0.5) * mainColorRadius
        if rc < min_rc:
            return None

        # Direção: diferença de vetores é invariante a translação, então
        # tanto faz usar coordenadas locais ou globais aqui -- mantido local
        # (idêntico ao cálculo original).
        direction = np.array([cx, -cy]) - np.array([x_m, -y_m])
        modDir = np.linalg.norm(direction)
        if modDir > 1e-6:
            direction = direction / modDir

        # Coordenadas GLOBAIS (somando o offset da janela): usadas na
        # conversão px->cm e em todo campo que é consumido fora desta janela
        # (desenho em self.frameResult, identificação na Parte 3, etc.).
        cx_g, cy_g = cx + x_offset, cy + y_offset
        x_m_g, y_m_g = x_m + x_offset, y_m + y_offset

        xcm, ycm = self.GetPointVirtual(self.TransformPoint(np.array([cx_g, cy_g])))

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
    def ResolveCollisionBlob(self, img_shape, cnt, cx, cy, r, n_est, winSize, timestamp,
                                x_offset=0, y_offset=0, imgHSV=None):
        """
        Separa um blob fundido (colisão) usando:
        1. Busca geométrica dirigida (tags T1/T2)
        2. Erosão sucessiva
        3. Clustering por cor (K‑means) – NOVO FALLBACK
        4. Predição do Kalman (último recurso)
        """
        centers = []

        # =========== PASSO 1: busca geométrica dirigida ==============
        if imgHSV is not None:
            geo_centers, _claimed_bots = self._GeometricCollisionSplit(
                imgHSV, cnt, cx, cy, r, n_est, winSize, timestamp, x_offset, y_offset
            )
            centers.extend(geo_centers)

        # =========== PASSO 2: erosão cega =============================
        if len(centers) < n_est:
            need = n_est - len(centers)
            eroded = self._SplitBlobByErosion(img_shape, cnt, cx, cy, need, winSize)
            for c in eroded:
                if all(np.hypot(c[0] - ex, c[1] - ey) > 0.5 * r for (ex, ey) in centers):
                    centers.append(c)
                if len(centers) >= n_est:
                    break

        # =========== PASSO 2.5 (NOVO): clustering por cor ============
        if len(centers) < n_est and imgHSV is not None:
            # Cria máscara do blob
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            # Recorta a região do blob para acelerar
            x, y, w, h = cv2.boundingRect(cnt)
            margin = int(0.3 * winSize)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img_shape[1], x + w + margin)
            y2 = min(img_shape[0], y + h + margin)
            roi_mask = mask[y1:y2, x1:x2]
            roi_hsv = imgHSV[y1:y2, x1:x2]
            pts = cv2.findNonZero(roi_mask)
            if pts is not None and len(pts) > 10:
                pts = pts.reshape(-1, 2)
                colors = roi_hsv[pts[:,1], pts[:,0]].astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, _ = cv2.kmeans(colors, n_est, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                for i in range(n_est):
                    mask_i = (labels == i).flatten()
                    pts_i = pts[mask_i]
                    if len(pts_i) > 5:
                        cx_cluster = np.mean(pts_i[:,0]) + x1
                        cy_cluster = np.mean(pts_i[:,1]) + y1
                        if all(np.hypot(cx_cluster - ex, cy_cluster - ey) > 0.4 * r for (ex, ey) in centers):
                            centers.append((cx_cluster, cy_cluster))
                            if len(centers) >= n_est:
                                break

        # =========== PASSO 3: Kalman puro (fallback final) ===========
        if len(centers) < n_est:
            need = n_est - len(centers)
            kalman_centers = (
                self._GetKalmanPredictedCenters(ID_Team.TEAM_ALLY, need, timestamp) +
                self._GetKalmanPredictedCenters(ID_Team.TEAM_ENEMY, need, timestamp)
            )
            kalman_centers.sort(key=lambda p: np.hypot(p[0] - cx, p[1] - cy))
            for (px, py) in kalman_centers[:need]:
                if np.hypot(px - cx, py - cy) < 2 * r:
                    if all(np.hypot(px - ex, py - ey) > 0.5 * r for (ex, ey) in centers):
                        centers.append((px, py))
                    if len(centers) >= n_est:
                        break

        return centers[:n_est]


    def _GeometricCollisionSplit(self, imgHSV, cnt, cx, cy, r, n_est, winSize, timestamp,
                                  x_offset=0, y_offset=0):
        """
        Helper privado da PARTE 2 (v2). Implementa o passo 1 (busca
        geométrica dirigida por robô conhecido) descrito no docstring de
        ResolveCollisionBlob -- é a peça que substitui a erosão como método
        PRIMÁRIO de separação do blob fundido.

        Ideia central: em vez de perguntar "quantas manchas dá pra separar
        aqui dentro" (erosão, cega quanto a identidade), pergunta "cada robô
        que eu já conheço a cor está aqui dentro?" -- um teste de hipótese
        por robô, igual ao já usado em DetectBotInRoi/SearchBot, só que
        repetido para todos os candidatos plausíveis dentro da MESMA ROI do
        blob fundido.

        `imgHSV` precisa estar no MESMO referencial LOCAL de `cnt`/`cx`/`cy`
        (o mesmo `imgHSV` recebido por DetectPlayerCandidates) -- ou seja,
        NÃO é necessariamente self.imgHSV/self.hsv_fieldReduce inteiro, a
        menos que a janela de detecção seja o campo inteiro com offset
        (0, 0), como é o caso hoje em DetectPlayers.

        Retorna (centers, claimed_bots):
            centers: lista de até `n_est` centros (x, y) LOCAIS a `cnt`.
            claimed_bots: set com os objetos Robot já resolvidos aqui (para
                uso futuro por quem quiser evitar retrabalho na Parte 3;
                não é consumido dentro desta função).
        """
        H, W = imgHSV.shape[:2]

        # --- ROI de trabalho: bounding box do blob + margem de ~1 robô ---
        x, y, w, h = cv2.boundingRect(cnt)
        margin = int(0.6 * winSize)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(W, x + w + margin)
        y2 = min(H, y + h + margin)
        if x2 <= x1 or y2 <= y1:
            return [], set()
        roi_hsv = imgHSV[y1:y2, x1:x2]

        # --- Gate de sanidade em pixels (equivalente ao MAX_JUMP_CM usado
        #     no resto do pipeline para gating de posição) ---
        max_jump_px = max(self.MAX_JUMP_CM * self.prop_px_cm, float(winSize))

        # --- Monta lista de robôs "elegíveis" (T1/T2 já conhecidos) -------
        # Aliados: cor sempre conhecida (config fixa), então são elegíveis
        # mesmo sem Kalman inicializado (robô que acabou de entrar em campo).
        # Inimigos: só elegíveis depois de identificados ao menos uma vez
        # (bot.colorCar1/colorCar2 cacheados por setColor em frames
        # anteriores) -- antes disso, caem no fallback de erosão/Kalman.
        candidates = []  # (priority, bot, team_hsv, primary_hsv, secondary_hsv, pred_local)
        for team, team_list in ((ID_Team.TEAM_ALLY, self.allyTeam),
                                 (ID_Team.TEAM_ENEMY, self.enemyTeam)):
            team_hsv = self.allyColor if team == ID_Team.TEAM_ALLY else self.enemyColor
            for bot in team_list:
                primary = getattr(bot, "colorCar1", None)
                secondary = getattr(bot, "colorCar2", None)
                if primary is None or secondary is None:
                    continue

                pred_local = None
                if getattr(bot, "kalman_initialized", False):
                    try:
                        x_pred, y_pred, _ = bot.predict(timestamp)
                        xg, yg = self.GetImageRealIndice((x_pred, y_pred))
                        pred_local = (float(xg) - x_offset, float(yg) - y_offset)
                    except Exception:
                        pred_local = None

                if pred_local is not None:
                    dist_to_blob = float(np.hypot(pred_local[0] - cx, pred_local[1] - cy))
                    # Só entra como candidato "provável" se o Kalman prevê
                    # esse robô dentro (ou bem perto) do próprio blob fundido.
                    if dist_to_blob > 2.5 * r + winSize:
                        continue
                    priority = dist_to_blob
                else:
                    # Sem Kalman (robô novo/cold-start): ainda elegível --
                    # testado depois dos que têm previsão, sem gate espacial
                    # prévio (o gate de cor + jump ainda se aplica ao "hit").
                    priority = 2.5 * r + winSize + 1.0

                candidates.append((priority, bot, team_hsv, primary, secondary, pred_local))

        if not candidates:
            return [], set()

        # Prioriza robôs cujo Kalman já aponta pra dentro do blob -- são os
        # melhores palpites de "quem colidiu aqui", testados primeiro.
        candidates.sort(key=lambda c: c[0])

        centers = []
        claimed_bots = set()
        used_local_points = []

        for priority, bot, team_hsv, primary, secondary, pred_local in candidates:
            if len(centers) >= n_est:
                break

            hit = self.LocateBotGeometric(roi_hsv, team_hsv, primary, secondary)
            if hit is None:
                continue  # T1/T2 desse robô não apareceram nesta ROI

            center_local = (float(hit['center'][0] + x1), float(hit['center'][1] + y1))

            # --- Gate de sanidade (ataca o caso "troca de ID por simetria
            # de cores"): um hit que bate na cor mas está longe demais de
            # onde ESTE robô deveria estar é rejeitado -- ele provavelmente
            # pertence a outro robô com cor parecida/trocada, que vai ser
            # encontrado (corretamente) em sua própria iteração.
            if pred_local is not None:
                jump = np.hypot(center_local[0] - pred_local[0], center_local[1] - pred_local[1])
                if jump > max_jump_px:
                    continue

            # --- Evita dois robôs "roubarem" quase o mesmo ponto físico ---
            if any(np.hypot(center_local[0] - ux, center_local[1] - uy) < 0.4 * r
                   for (ux, uy) in used_local_points):
                continue

            centers.append(center_local)
            used_local_points.append(center_local)
            claimed_bots.add(bot)

        return centers, claimed_bots

    def _SplitBlobByErosion(self, img_shape, cnt, cx, cy, n_est, winSize):
        """
        Helper privado da PARTE 2. Aplica erosão sucessiva na máscara do
        próprio blob (apenas a região do contorno) até que o número de
        componentes conectados seja >= n_est. Retorna lista de centros
        (x, y) no MESMO referencial local de `cnt`.
        """
        # Cria máscara da região do blob (apenas o contorno)
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        # Pega o bounding box para limitar as operações (otimização)
        x, y, w, h = cv2.boundingRect(cnt)
        margin = int(0.3 * winSize)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_shape[1], x + w + margin)
        y2 = min(img_shape[0], y + h + margin)

        # Recorta a máscara para a ROI do blob
        roi_mask = mask[y1:y2, x1:x2]
        if roi_mask.size == 0:
            return []

        # Kernel de erosão (pequeno, para não perder a forma)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        current = roi_mask.copy()
        centers = []
        max_iter = 25
        min_area = 5  # área mínima para considerar um componente válido

        for i in range(max_iter):
            # Encontra componentes conectados (8-conectividade)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                current, connectivity=8
            )
            # Ignora o fundo (label 0)
            valid = []
            for l in range(1, num_labels):
                area = stats[l, cv2.CC_STAT_AREA]
                if area >= min_area:
                    valid.append(l)
            if len(valid) >= n_est:
                # Pega os centroides dos primeiros n_est componentes
                # (ordem arbitrária, mas consistente)
                for l in valid[:n_est]:
                    cx_roi = centroids[l][0] + x1
                    cy_roi = centroids[l][1] + y1
                    centers.append((cx_roi, cy_roi))
                break
            # Erosão
            current = cv2.erode(current, kernel, iterations=1)
            if cv2.countNonZero(current) == 0:
                break
        else:
            # Se não separou, usa o que tem (mas pode ser insuficiente)
            if centers:
                pass  # já temos alguns centros
            else:
                # Fallback: pega o centroide do blob inteiro (um só)
                centers.append((cx, cy))

        return centers

    def _GetKalmanPredictedCenters(self, team, n_needed, timestamp):
        """
        Helper privado da PARTE 2. Retorna até n_needed centros PREVISTOS
        pelo Kalman (predict puro, sem corrigir/atualizar o filtro) para
        robôs do time indicado que já têm o Kalman inicializado.

        ATENÇÃO (preservado fielmente do código original): o ponto retornado
        vem de self.GetImageIndice(...), que converte cm -> pixels da IMAGEM
        VIRTUAL (ver docstring de GetImageIndice), enquanto `cx`/`cy` em
        ResolveCollisionBlob estão no referencial local da janela de
        detecção (fieldReduce ou menor). Ou seja, a comparação de distância
        feita em ResolveCollisionBlob já misturava esses dois referenciais
        na versão original -- mantive exatamente assim para não alterar o
        comportamento numérico atual, mas é o primeiro ponto a revisar
        quando for ajustar esta função com mais cuidado.
        """
        team_list = self.allyTeam if team == ID_Team.TEAM_ALLY else self.enemyTeam
        centers = []
        for bot in team_list:
            if bot.kalman_initialized:
                # Prediz para o timestamp atual (não altera o filtro)
                x_pred, y_pred, _ = bot.predict(timestamp)
                # Converte para coordenadas de imagem (pixels)
                xi, yi = self.GetImageIndice((x_pred, y_pred))
                centers.append((xi, yi))
                if len(centers) >= n_needed:
                    break
        return centers

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

    def DetectAllyRobot(self, window, colorP, colorS):
        """
        Verifica se há um robô aliado dentro de uma janela, com base em duas cores (primária e secundária).

        Retorna:
            bool: True se ambas as cores forem detectadas, False caso contrário.

        Parâmetros:
            window (np.ndarray): imagem em que será feita a busca.
            colorP (list[int]): cor primária em HSV (ex: [H, S, V]).
            colorS (list[int]): cor secundária em HSV (ex: [H, S, V]).
        """
        try:
            # Cria limites HSV das duas cores
            p_lower, p_upper = self.CreateColorBounds(colorP)
            s_lower, s_upper = self.CreateColorBounds(colorS)

            # Obtém contornos das duas cores na janela
            contoursP = self.FindBinaryContours(window, p_lower, p_upper)
            contoursS = self.FindBinaryContours(window, s_lower, s_upper)

            # Define o raio mínimo esperado (pré-calculado para evitar recomputação)
            min_radius = max(2, 0.08 * self.secColorRadius)

            # Verifica se existem contornos com tamanho significativo
            primaryFound = any(cv2.minEnclosingCircle(c)[1] >= min_radius for c in contoursP)
            secondaryFound = any(cv2.minEnclosingCircle(c)[1] >= min_radius for c in contoursS)

            return primaryFound and secondaryFound

        except Exception as e:
            print(f"[detect_ally_robot] Erro ao processar imagem: {e}")
            return False

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

    def SearchBot(self, img, roi, team: ID_Team, bot_id: ID_Robots, timestamp, debug=False):
        """
        Procura um robô específico na ROI seguindo o MESMO fluxo geométrico das
        tags (Tag_Time + T1 + T2) usado por DetectBotInRoi. Primeiro tenta o
        robô-alvo (bot_id); se ele não bater a hipótese geométrica em nenhum blob
        da cor do time, tenta os demais robôs do mesmo time dentro da mesma ROI
        (trocando apenas as cores T1/T2 de cada candidato), e os atualiza também.

        NOTA: esta função não é chamada em nenhum lugar do pipeline atual (FilteredDetection
        usa SearchBots -> DetectBotInRoi, não esta). Ela foi mantida e alinhada com o mesmo
        gating de distância (self.MAX_JUMP_CM) e a mesma regra de inicialização do Kalman
        (setPosition antes de updatePosition) usadas em DetectPlayers e em FilteredDetection,
        para o caso de vir a ser reaproveitada como busca avulsa por um robô específico.
        """
        bot = self.GetBotById(team, bot_id)
        if bot is None:
            if debug:
                print(f"[search_bot] Bot {team, bot_id} não existe na lista")
            return False

        x0, y0, w0, h0 = roi
        # --------------------------------------------------------
        # Valida ROI
        # --------------------------------------------------------
        if w0 <= 0 or h0 <= 0 or x0 < 0 or y0 < 0 or x0+w0 > img.shape[1] or y0+h0 > img.shape[0]:
            if debug:
                print(f"[search_bot] ROI inválida: {roi}")
            return False

        # --------------------------------------------------------
        # Recorte da ROI e conversão HSV local
        # --------------------------------------------------------
        window = img[y0:y0+h0, x0:x0+w0]
        windowHSV = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)

        team_hsv = self.allyColor if team == ID_Team.TEAM_ALLY else self.enemyColor
        teamBots = self.allyTeam if team == ID_Team.TEAM_ALLY else self.enemyTeam

        winSize = int(18 * self.prop_px_cm)
        half_win = winSize // 2
        playerRadius = (self.TAG_L_CM / 2.0) * np.sqrt(2) * self.prop_px_cm

        # --------------------------------------------------------
        # Helper: alimenta o Kalman com o mesmo fail-safe de DetectPlayers/
        # FilteredDetection (gating de distância + setPosition antes de
        # updatePosition quando o filtro ainda não está inicializado).
        # Retorna False se o candidato foi rejeitado por gating.
        # --------------------------------------------------------
        def _feed_kalman(target, x_cm, y_cm, dir_vec, bot_window):
            if target.kalman_initialized:
                last_pos = getattr(target, "position_filtered", None)
                if last_pos is not None:
                    jump = float(np.hypot(x_cm - last_pos[0], y_cm - last_pos[1]))
                    if jump > self.MAX_JUMP_CM:
                        if debug:
                            print(f"[search_bot][GATING] {target.team, target.id}: "
                                  f"salto de {jump:.1f}cm rejeitado.")
                        return False
                target.updatePosition(x=x_cm, y=y_cm, direction=dir_vec, image=bot_window, time=timestamp)
            else:
                target.setPosition(x_cm, y_cm, dir_vec, bot_window, time=timestamp)
            return True

        # --------------------------------------------------------
        # Helper: aplica um "hit" geométrico (passos 8-9) a um robô-alvo,
        # convertendo para cm, alimentando o Kalman e atualizando os binários
        # globais/desenho de debug.
        # --------------------------------------------------------
        def _report(target_bot, hit):
            center_local = hit['center']
            direction_img = hit['direction']
            cnt = hit['contour']

            xi_global = int(round(center_local[0] + x0))
            yi_global = int(round(center_local[1] + y0))

            direction = np.array([direction_img[0], -direction_img[1]], dtype=float)
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])

            xi_local, yi_local = center_local[0], center_local[1]
            x1l = max(0, int(xi_local - half_win))
            y1l = max(0, int(yi_local - half_win))
            x2l = min(w0, int(xi_local + half_win))
            y2l = min(h0, int(yi_local + half_win))
            bot_window = window[y1l:y2l, x1l:x2l]

            xv, yv = self.TransformPoint(np.array([xi_global, yi_global]))
            xcm, ycm = self.GetPointVirtual(np.array([xv, yv]))

            if not _feed_kalman(target_bot, xcm, ycm, direction, bot_window):
                return False

            target_bot.updtPositionImg(xi_global, yi_global, playerRadius)
            target_bot.setStatus(True)
            self.DrawPlayerCircle(self.frameResult, target_bot)
            self.DrawPlayerVirtual(target_bot)

            cont_global = cnt.copy()
            cont_global[:, 0, 0] += x0
            cont_global[:, 0, 1] += y0
            if self.binaryPlayers is not None:
                cv2.drawContours(self.binaryPlayers, [cont_global], -1, 255, -1)
            if team == ID_Team.TEAM_ALLY and self.binaryAllies is not None:
                cv2.drawContours(self.binaryAllies, [cont_global], -1, 255, -1)
            return True

        # --------------------------------------------------------
        # 1) Tenta localizar o robô-alvo diretamente (fluxo geométrico completo)
        # --------------------------------------------------------
        hit = self.LocateBotGeometric(windowHSV, team_hsv, bot.colorCar1, bot.colorCar2)
        if hit is not None and _report(bot, hit):
            if debug:
                print(f"✅ Detectado Robô {team.name} {bot_id.name} (específico) na ROI {roi}")
            return True

        # --------------------------------------------------------
        # 2) Fallback: procura outros robôs do mesmo time na mesma ROI
        # (mesma Tag_Time, cores T1/T2 diferentes por candidato)
        # --------------------------------------------------------
        found = False
        for other_bot in teamBots:
            if other_bot.id == bot_id:
                continue
            hit = self.LocateBotGeometric(windowHSV, team_hsv, other_bot.colorCar1, other_bot.colorCar2)
            if hit is None:
                continue
            if _report(other_bot, hit):
                found = True
                if debug:
                    print(f"✅ Detectado Robô {team.name} {other_bot.id.name} (fallback) na ROI {roi}")

        if debug and not found:
            print(f"[search_bot] Nenhum robô do time {team.name} localizado geometricamente na ROI {roi}")

        return found
    
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

    def PredictBotFallback(self, bot, timestamp, mark_detected=False):
        """
        Atualiza o objeto bot usando a predição do Kalman sem alterar o filtro.
        - bot: instância de Robot
        - timestamp: tempo atual (mesmo que passou para predict_with_cov)
        - mark_detected: se True marca bot.detected = True (útil se quiser desenhar)
        """
        try:
            st_pred, P_pred = bot.predict_with_cov(timestamp)  # (6x1), (6x6)
            x_pred = float(st_pred[0, 0])
            y_pred = float(st_pred[1, 0])
            theta_pred = float(st_pred[2, 0])

            # Use a API do Robot para atualizar sem tocar no Kalman
            # signature: SetPosition(x, y, theta, timestamp, image=None)
            bot.SetPosition(x_pred, y_pred, theta_pred, timestamp, image=None)

            # status: por padrão fallback não é "detectado" (mas escolha sua política)
            bot.detected = bool(mark_detected)

            # update bbox/view já feito por SetPosition
            # opcional: atualizar imagem virtual (ponto) para debug
            if hasattr(self, "virtualImg"):
                xi, yi = self.GetImageIndice((x_pred, y_pred))
                bot.updtPositionImg(int(xi), int(yi), int(bot.radius * 3))  # ri visual aproximado

            return True
        except Exception as e:
            print(f"[VS][_predict_bot_fallback] erro ao aplicar fallback para bot {getattr(bot,'id', '?')}: {e}")
            return False

    def PredictBallFallback(self, timestamp, mark_detected=False):
        """
        Predição fallback para a bola usando o Kalman da bola.
        Usa ball.predict/predict_with_cov e atualiza com SetPosition da Ball.
        """
        try:
            # ball tem predict_with_cov? seu Ball tem predict(timestamp) (retorna x,y,theta)
            # se tiver predict_with_cov, prefira-o para obter P_pred; aqui usamos predict_with_cov se existir
            if hasattr(self.ball, "predict_with_cov"):
                st_pred, P_pred = self.ball.predict_with_cov(timestamp)
                x_pred = float(st_pred[0, 0])
                y_pred = float(st_pred[1, 0])
                theta_pred = float(st_pred[2, 0])
            else:
                x_pred, y_pred, theta_pred = self.ball.predict(timestamp)

            # converter predição virtual para CM já está no estado do filtro (x,y são cm)
            # usar API da bola que você definiu: SetPosition(x, y, r, timestamp=0.0, theta=None)
            rb = getattr(self, "ballRadiusP", self.ball.radius)
            self.ball.SetPosition(x_pred, y_pred, rb, timestamp, theta=theta_pred)

            self.ball.detected = bool(mark_detected)

            # debug: atualizar pos em imagem real
            if hasattr(self, "virtualImg"):
                xi, yi = self.GetImageIndice((x_pred, y_pred))
                try:
                    self.ball.setImgPosition(int(xi), int(yi), int(rb / self.prop_px_cm))
                except Exception:
                    pass

            return True
        except Exception as e:
            print(f"[VS][_predict_ball_fallback] erro: {e}")
            return False
        
    def HandleRobotLoss(self, bot, team, timestamp):
            """Gerencia falha de detecção: Predição Suave ou Reset."""
            key = (bot.id, team)
            curr_missed = self.missed_frames_robots.get(key, 0) + 1
            self.missed_frames_robots[key] = curr_missed

            if curr_missed <= self.MAX_MISSED_FRAMES_ROBOT:
                # Fallback: Predição do EKF (mantém movimento suave)
                # bot.predict(t) retorna (x, y, theta) preditos pelo modelo
                pred_x, pred_y, pred_theta = bot.predict(timestamp)
                
                # Atualiza visualmente sem tocar no estado do filtro
                bot.SetPosition(pred_x, pred_y, pred_theta, timestamp)
                bot.setStatus(True) # Mantém visualmente ativo
            else:
                # Perda Real: Desliga e marca para reset
                bot.setStatus(False)
                self.robot_kalman_reset_flags[key] = True

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
                if debug: print(f"[VS][PROC] Falha ao usar cache do campo: {e}. Forçando detecção.")
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
                    if debug: print(f"[VisionSystem] Erro no Tracking: {e}. Reiniciando detecção.")
                    
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
        print("[VisSys][FilterdDetection] Estou trabalhando aqui")
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
                if debug: print(f"[filtered_detection] Campo muito pequeno. Resetando.")
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
                                print(f"[FilteredDetection][GATING] Robô {key}: salto de {jump:.1f}cm rejeitado.")
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
                    print(f"[filtered_detection][ERROR robot {key}]: {e}")
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
    print("Utilizada em função de main")