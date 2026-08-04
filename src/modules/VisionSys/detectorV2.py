# ==========================================================================================
# MÓDULO DE FUNÇÕES PARA ALGORÍTMO DE DETECÇÃO VSS (version v3.2)
#==========================================================================================
'''
    @GNOMIO: Sismtea de detecção de objetos VSS (Vision System Soccer) versão 3.2.13   
    
    Versão: v3.2.13
    Última modificação: 20/07/2026
    Autor: Saulo (update)

    Patch Notes v3.2.13:
    - Resolvido algumas incongruências no código que estavam afetando a eficiência

    Obs: Ainda está numa versão BETA, necessário testes para verificar se está
    corretamente funcionando!!!

'''
#importando bibliotecas necessárias para o código
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

# ====================== DEFINIÇÕES DAS CLASSES DE OBJETO DO SISTEMA ==================
#======================|| Sistema de detecção POO||======================================#
#Classe principal do sistema de visão que irá executar as funções
class VisionSystem:
    '''
    É a classe principal do sistema de visão. Ela é responsável por processar uma imagem
    que pode vim de vários meios e extrair as informações pertinentes que serão utilizadas
    no sistema de visão.

    Ela funciona através de configurações enviadas pelo emulador, um objeto de captura e informar
    se vai ou não utilziar a GPU para agilizar o processamento.
    '''
    #Inicializando objeto do sistema de detecção
    def __init__(self,config: EConfig = EConfig(), debug:bool = False, capture: Capture = None, UseCuda:bool = False, GPUType:GPUType = None):
        '''
            Inicializando o sistema de visão para realizar a captura de dados e tradução.
            Para isso é necessário passar as configurações do emulador (EConfig), o objeto de captura
            (capture) se ele tem ou não cuda  (useCuda) e qual o tipo de GPU (GPUType).
        '''
        #Configurando objetos
        self.CreateObjs()

        self._countProcess = 0

        self.bmk = Benchmark()

        # ===============================================================================
        # WATCHDOG CONFIG
        self.MAX_MISSED_FRAMES_BALL = 10  # ~0.3s a 30fps
        self.MAX_MISSED_FRAMES_ROBOT = 15 # Robôs tem mais inércia, toleramos mais
        
        # ESTADO
        self.missed_frames_ball = 0
        self.ball_kalman_reset_flag = False # Começa resetado
        
        # Para robôs, usamos um dicionário {robot_id: missed_count}
        # Inicializa com 0 para todos os seus robôs (ex: 3 aliados, 3 inimigos)
        self.missed_frames_robots = {} 
        self.robot_kalman_reset_flags = {}
        #+====================================================================================

        #variável que me dirá quantas vezes o sistema de visão foi chamado
        self._count: int            = 0 
        ''' Variável responsavel por dizer quantas vezes foi executado o sistema '''
        self._firstTimeExec: int    = 0 
        ''' Variável que diz quanto tempo se passou desde a primeira execução do código'''
        #Carregando as configurações do sistema de visão
        self.config:EConfig    = config
        
        #objeto de captura internas
        self._capture:Capture  = capture

        #verifica se existe suporte ao CUDA
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

        '''
        @GNOMIO: Caso queira utilizar esses dados no FiraSIM, utilize esses
        valores abaixo. Eles são do centro exato na imagem virtual (EM PIXELS!)
        '''
        #self.xnv = int(10*3+67+(150*3)/2)
        #self.ynv = int(402 - 130*3/2)

        #self.coordOrigin = np.array([67,402])
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



        #Variáveis internas do sistema de visão que serão importantes para o processamento
        #Configurações
        self.offSetWindow       = 10                 
        '''Tamanho extra de janela utilizada'''
        self.offSetErode        = 0                    
        '''Quantidade padrão de erosões na imagem'''
        self.dimMatrix          = 25                     
        '''Dimensão da matriz de convolução na imagem'''
        self.Thrashhold         = 235                   
        '''Limiar de binarização da imagem'''
        self.pixelWidth         = 1
        '''Tamanho de um pixel normal'''

        #Imagens
        self.frameOrigin        = None             
        '''responsável por guardar a imagem do campo'''
        self.ballImg            = None 
        ''' Imagem da bola que é utilizada para processar e procurar os jogadores'''
        self.fieldReduce        = None              
        '''Imagem do campo reduzida'''
        self.frameResult        = None              
        '''Imagem final já reduzida e processada'''
        self.imgReduce          = None              
        '''Imagem reduzida para utilizar no processamento'''
        self.binaryAllTeam      = None 

        #Imagem virtual que será utilizada para o processamenot
        self.virtual            = cv2.imread("src/data/images/CampoVirtual.png")
        self.virtualImg         = self.virtual.copy()   #As manipulações da imagem serão feitas nessa aqui

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
    
    # Implementação da lógica de processamento para várias coisas
    # ============ GRUPO A - MÉTODOS DE CONTROLE DA CLASSE ============
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
        # 3) RENDERIZAÇÃO / VISUALIZAÇÃO (O GARGALO REAL)
        # ===========================
        # AQUI está o segredo da performance. Só gastamos CPU desenhando se alguém for ver.
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

    # Funções auxiliares
    def SafeCall(self, func, *args, name="", **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[VS][SAFE_CALL] Erro ao executar {name}: {e}")
            traceback.print_exc()
            return None

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


    def ResetExecutionState(self):
        """
        Reseta apenas variáveis temporárias entre execuções do processamento.
        Preserva dados dos robôs, bola, campo e configurações permanentes.
        
        Deve ser chamado APÓS cada Proc() completo para limpar estado interno.
        """
        # ==================== IMAGENS TEMPORÁRIAS ====================
        self.frameOrigin = None
        self.fieldReduce = None
        self.frameResult = None
        self.imgReduce = None
        self.ballImg = None
        
        # ==================== IMAGENS BINÁRIAS ====================
        self.binaryObjects = np.zeros((1, 1), dtype=np.uint8)
        self.binaryPlayers = np.zeros((1, 1), dtype=np.uint8)
        self.binaryBall = np.zeros((1, 1), dtype=np.uint8)
        self.binReduceField = np.zeros((1, 1), dtype=np.uint8)
        self.binField = np.zeros((1, 1), dtype=np.uint8)
        self.binaryAllTeam = np.zeros((1, 1), dtype=np.uint8)
        
        # ==================== BUFFERS DE PROCESSAMENTO ====================
        self.playersWindows = [None, None, None, None, None, None]
        self.alliesWindows = [None, None, None]
        self.enimiesWindows = [None, None, None]
        
        # ==================== CONTADORES TEMPORÁRIOS ====================
        self.playersCount = 0
        self.alliesCount = 0
        self.enemiesCount = 0
        self.fieldDetectionFailCount  = 0
        # ==================== ESTADO DE DETECÇÃO ATUAL ====================
        # Reset apenas dos flags de detecção do frame atual
        # (não afeta histórico de posições)
        for bot in self.allyTeam:
            bot.detected = False  # Apenas o status de detecção atual
            bot.possessionBall = False  # Posse de bola é temporária
            
        for bot in self.enemyTeam:
            bot.detected = False
            bot.possessionBall = False
            
        if hasattr(self.ball, 'detected'):
            self.ball.detected = False
        
        # ==================== VARIÁVEIS DE TEMPO DO FRAME ====================
        # Não resetar: self.lastMajorTime, self.currentTime, self.dT
        # (são necessárias para controle temporal entre execuções)
        
        # ==================== RESET DA IMAGEM VIRTUAL ====================
        if hasattr(self, 'virtual'):
            self.virtualImg = self.virtual.copy()
        
        # ==================== LIMPEZA DE THREADS FINALIZADAS ====================
        # Limpa threads que já terminaram, mantém executor ativo
        self._threads = [t for t in self._threads if t.is_alive()]
        
        # Reseta a árvore de cores
        self.colorTree.clear()
        self.SetTreeColorDefault()

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

        if self.debug:
            self.DrawAllRobots()
            self.DrawBallDebug()
        return self.frameResult

    #Puxando as imagens de debug
    def GetDebugImages(self):
        return self.binaryObjects, self.binaryBall, self.binaryPlayers, self.binaryAllTeam 
        
        
    #Inicializando os objetos do sistema
    def CreateObjs(self):
        '''
            @GNOMIO: função responsável por criar os objetos do sistema de visão.
            Sendo eles a bola, os robôs, o campo e as áreas do campo.

            Esses objetos são inicializados sem informação, e só são cadastrados dps
        '''
        #Construção dos objetos necessários para realizar a análise
        #Objeto da bola
        self.ball = Ball()
        
        #Objeto dos robôs,
        #robôs aliados
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

    # atribuindo os valores dos pontos do campo
    def BuildField(self):
        '''
            Atribuindo os valores do campo que já foram carregados no campo

            Necessário converter dados para cm e nas coordenadas de O' (xn,yn)
        '''
        #adicionando extremidades virtuais 
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


    #extrair dados vindos do emulador 
    def ToMineData(self):
        '''
        Essa função extrai as informações vindas do Emulador no objeto EConfig.
        '''
        print("[VisionSystem]: Configurações carregadas")
        #puxando valores do objeto de configuração
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

        #Setando
        self.SetTreeColorDefault()
        # Atribuindo cores a arvore da decisão

    #definir novas configurações
    def SetConfigEmulator(self, config:EConfig):
        '''
            Essa função seta uma nova configuração para o sistema de visão pelo emulador
        '''
        self.config = config
        self.ToMineData()

    def SetTreeColorDefault(self):
        '''Setando configurações padrões da árvore de cores'''
        # Time Aliado
        self.colorTree.add_robot(
            ID_Team.TEAM_ALLY,      # time
            ID_Robots.ROBOT_ALLY_GOAL,  # ID do robô
            self.allyColor,         # cor principal do time
            self.goalAllyColor1,    # cor primária do robô
            self.goalAllyColor2     # cor secundária do robô
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
    #método para retornar o processamento
    def GetViewCapture(self):
        '''
            Retorna a janela de interesse do sistema de visão
        '''
        return self.viewCapture
    
    #definindo modo para resetar configurações
    def ResetVs(self):
        """Reset COMPLETO do sistema de visão"""
        
        # 1. Recria objetos principais
        self.CreateObjs()
        
        # 2. Reset de TODAS as variáveis de estado
        self._countProcess = 0
        self._count = 0
        self._firstTimeExec = 0
        self.lastMajorTime = 0
        self.currentTime = 0
        self.dT = 0

        # Salva caches dos inimigos
        self._enemy_last_pos = {}
        self._ally_last_pos = {}

        # 3. Reset de transformações
        self.homography_matrix = None
        self.inv_homography_matrix = None
        self.prop_px_cm = 1
        self.min_diag = (7.5 / 4) * np.sqrt(2) * self.prop_px_cm
        
        # 4. Reset de configurações de GPU/CPU
        self.GPUimg = None
        self.CPUimg = None
        self.emulatorMode = MODE_IMAGE
        
        # 5. Reset de imagens e buffers
        self.frameOrigin = None
        self.ballImg = None
        self.fieldReduce = None
        self.frameResult = None
        
        self.ResetExecutionState()
        
        # 7. Reconstroi o campo
        self.BuildField()

        # 8. Recaptura cores
        self.SetTreeColorDefault()

    #função para retornar homografia entre dois sistemas de pontos
    def GetHomographyMatrix(self, ptsSrc, ptsFinal):
        '''
            Essa função retorna a matrix 3x3 de homografia entre dois planos
            nesse caso, a imagem de detecção e uma imagem virtual. Nesse caso, a matrix vai mapear os pontos de entrada nos pontos de saída.

            Obs: A imagem de saída tem as dimensões de 645 por 413 px, e a proporção px/cm = 3
            Além disso, já foi configurado para esse calculo ser realizado com os pontos extremos do campo.

            Essa matrix fica salva no sistema de visão para poder realizar as devidas operações
            
            ### Variáveis
            - ptsSrc: conjunto de quatro pontos de entrada da imagem original
            - ptsFinal: Conjunto de pontos correspondentes na imagem final

            # Necessário que essas variáveis sejam vetores array.
        '''
        ptsSrc = np.array(ptsSrc, dtype='float32')
        ptsFinal = np.array(ptsFinal, dtype='float32')

        self.homography_matrix, _   =   cv2.findHomography(ptsSrc, ptsFinal)

        if self.homography_matrix is not None and np.linalg.cond(self.homography_matrix) < 1 / np.finfo(self.homography_matrix.dtype).eps:
            self.inv_homography_matrix  =   np.linalg.inv(self.homography_matrix)
        else:
            # A matriz de homografia é singular e não pode ser invertida
            self.inv_homography_matrix = np.eye(3)  # Matriz identidade 3x3
            print("[ERROR]: Matriz de homografia singular e foi substituída por uma matriz identidade.")

    #Transformar valores 
    def TransformPoint(self, ptSrc):
        '''
            Ela utiliza a matrix de homografia para transformar um ponto da imagem original num ponto da imagem
            virtual. Realizando essa conversão é possível saber uma boa aproximação, e desconsidera as distorções.
        '''

        ptSrc = np.array([[[ptSrc[0], ptSrc[1]]]], dtype=np.float32)
        ponto_transformado = cv2.perspectiveTransform(ptSrc, self.homography_matrix)
        return ponto_transformado[0][0]

    #aplica transformação inversa no ponto para recuperar o valor
    def InvTransformPoint(self, ptSrc):
        '''
            Realiza o trabalho inverso no TransformPoint, retornando para o espaço original da imagem.
        '''
        ptSrc = np.array([[[ptSrc[0], ptSrc[1]]]], dtype=np.float32)
        ponto_transformado = cv2.perspectiveTransform(ptSrc, self.inv_homography_matrix)
        return ponto_transformado[0][0]
        
    #Passa os indices da matrix final e transforma em valores em cm
    def GetPointVirtual(self, ptSrc):
        '''
            Com o "ptSrc" da imagem virtual é encontrado sua posição em relação ao
            novo sistema de coordenadas O', esse valor em cm será o que será guardado nos objetos
            (bola, robô e campo). Assim será possível realizar os calculos apenas no mundo virtual

            Deve-se passar o índice correspondente a imagem virtualizada e ele retorna
            a posição com cm (m/100)

            O resultado é um ponto em centímetros (cm)
        '''

        #   transforma o ponto no novo sistema de coordenadas
        
        x = ptSrc[0]
        y = ptSrc[1]

        #   coordenada final
        x_f = (x - self.xnv)/3
        y_f = (self.ynv - y)/3

        #caso a função seja utilizada num objeto Point2D, ela funciona assim:
        return x_f,y_f

    #com a posição em O' (em cm), transforma num índice na imagem:
    def GetImageIndice(self,ptSrc):
        '''
            Pega o valor do ponto em cm, e transforma em índices para a imagem virtual
            para poder, então desenhar-lo.
        '''
        x_f =int(ptSrc[0]*3 +self.xnv)
        y_f = int(self.ynv-ptSrc[1]*3)

        return x_f, y_f

    #definindo uma função para retornar a coordenada na imagem reduzida
    def GetImageRealIndice(self, ptSrc):
        '''
            Esse método pega as coordenadas virtuais em O' e passa retorna para os indices da imagem real.
        '''
        #pego os valores dos indices na imagem virtual
        x_i, y_i = self.GetImageIndice(ptSrc)
    
        #pego os valores dos indices na imagem virtual e aplica a homografia inversa
        #retornando a imagem reduzida
        return self.InvTransformPoint([x_i, y_i])



    def GetObjects(self):
        '''
            Função responsável por retornar os objetos do sistema de visão

        '''
        #construindo dicionário com os objetos 
        objects = {
            ID_Objects.ALLIES:self.allyTeam,
            ID_Objects.ENEMIES: self.enemyTeam,
            ID_Objects.BALL:self.ball,
            ID_Objects.FIELD:self.field,
            'timestamp': self.dT 
        }

        return objects 
    
    def GetFrameProtobuff(self):
        """
        Gera o pacote Protobuf (Frame) com dados filtrados pelo Kalman.
        Realiza a conversão de unidades (cm -> m) e de cinemática (Rodas -> Global).
        """
        frame = common_pb2.Frame()

        # =========================================================================
        # 1. BOLA (Estado: x, y, theta, vx, vy)
        # =========================================================================
        if self.ball is not None:
            try:
                # --- Posição ---
                # Prioriza o valor filtrado se disponível
                if hasattr(self.ball, 'position_filtered') and self.ball.position_filtered is not None:
                    bx, by = self.ball.position_filtered
                else:
                    bx, by = self.ball.position
                
                # --- Velocidade ---
                # Se a bola tem Kalman, o estado já é [vx, vy]
                bvx, bvy = 0.0, 0.0
                if hasattr(self.ball, 'velocity_filtered') and self.ball.velocity_filtered is not None:
                    # Assume que o filtro da bola retorna [vx, vy] diretamente
                    vel = self.ball.velocity_filtered
                    bvx, bvy = vel[0], vel[1]
                
                # Preenchimento (Convertendo cm -> metros)
                frame.ball.x = float(bx) / 100.0
                frame.ball.y = float(by) / 100.0
                frame.ball.z = 0.0
                frame.ball.vx = float(bvx) / 100.0
                frame.ball.vy = float(bvy) / 100.0
                frame.ball.vz = 0.0
            except Exception as e:
                if self.debug: print(f"[VS] Erro Bola Protobuff: {e}")

        # =========================================================================
        # 2. HELPER: ROBÔS (Estado: x, y, theta, vL, vR, omega)
        # =========================================================================
        def fill_robot_proto(source_bot, proto_bot):
            # --- Posição e Orientação ---
            # O @property position_filtered já trata se o Kalman está init ou não
            rx, ry = source_bot.position_filtered
            r_theta = source_bot.theta_filtered # Radianos

            # --- Velocidade (O Pulo do Gato) ---
            # O estado do seu Kalman é [x, y, th, vL, vR, w]
            # O Protobuf quer [vx, vy] (Global)
            
            v_wheels = source_bot.velocity_filtered # Retorna np.array([vL, vR])
            vL = v_wheels[0]
            vR = v_wheels[1]
            r_omega = source_bot.omega_filtered

            # Conversão: Cinemática Diferencial -> Velocidade Linear Global
            # V_linear_robô = (vR + vL) / 2
            v_lin = (vR + vL) / 2.0

            # Projeção no eixo global (X, Y)
            r_vx = v_lin * np.cos(r_theta)
            r_vy = v_lin * np.sin(r_theta)

            # --- Preenchimento do Pacote (cm -> m) ---
            proto_bot.robot_id = int(source_bot.id.value) if hasattr(source_bot.id, 'value') else int(source_bot.id)
            proto_bot.x = float(rx) / 100.0
            proto_bot.y = float(ry) / 100.0
            proto_bot.orientation = float(r_theta)
            
            # Velocidades convertidas
            proto_bot.vx = float(r_vx) / 100.0
            proto_bot.vy = float(r_vy) / 100.0
            proto_bot.vorientation = float(r_omega)

        # =========================================================================
        # 3. POPULANDO OS ROBÔS
        # =========================================================================
        
        # Time Aliado -> Yellow (Seguindo sua convenção)
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
    #===============| Definindo funções básicas|==============================
    #puxando imagem
    # ============ GRUPO B - MÉTODOS BÁSICOS DE PROCESSAMENTO DA IMAGEM ============
    def LoadImage(self, imgPath):
        '''
            Função que carrega imagem na CPU por meio de um caminho (imgPath). Sem usar a GPU
        '''
        self.imgOrigim = cv2.imread(imgPath) 
    
    #transformando imagem em tons de cinza
    def GrayScale(self,img):
        '''
            Coloca a imagem passada em tons de cinza, Sem usar a GPU
        '''
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #aplicando o filtro mediana para possíveis ruídos
    def MedianBlur(self, img, kernelSize = 3):
        '''
            Aplica o filtro de mediana para reduzir ruídos. Sem usar a GPU.
        '''
        return cv2.medianBlur(img, kernelSize)
    
    #binarizando a imagem indo de um limir até 255
    def BinarizeUp(self, img, threshold=150):
        '''
            Binariza a imagem por meio de um threshold, ou seja, um limiar
            de intensidade dos pixels. Para isso, a imagem tem que estar em
            tons de cinza. Tratada pela função MedianBlur().
        '''
        _,bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return bin

    #trata ruídos da imagem binarizada
    def TraitNoise(self, binImg, it=1):
        """
        Reduz ruídos em imagens binarizadas.

        Parâmetros:
            binImg (np.ndarray): imagem binarizada (grayscale ou binária 0/255).
            it (int): número de iterações de erosão/dilatação (default = 1).

        Retorna:
            np.ndarray: imagem binarizada com menos ruído.
        """
        try:
            # Garante tipo binário
            if binImg.dtype != np.uint8:
                binImg = cv2.convertScaleAbs(binImg)

            # Estrutura cruzada pequena remove ruídos pontuais sem deformar objetos
            kernel = self.kernel_morph

            # Erosão seguida de dilatação = abertura morfológica (remove ruídos brancos)
            binImgProc = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel, iterations=it)

            # (Opcional) fechamento remove buracos pequenos dentro dos objetos
            binImgProc = cv2.morphologyEx(binImgProc, cv2.MORPH_CLOSE, kernel, iterations=max(1, it - 1))

            return binImgProc

        except Exception as e:
            print(f"[SystemVision][TRAIT_NOISE]: Erro ao tratar ruído — {e}")
            return binImg

    
    #recuperando objeto de maior área
    def GetObject(self,img):
        '''
            Retorna o maior contorno encontrado ou None se não houver contornos.
        '''
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)
    
    #recuperando coordenadas extremas que englobam o objeto maior 
    def GetPerspective(self, contour):
        """
        Retorna as coordenadas (x1, y1, x2, y2) do retângulo delimitador
        do contorno fornecido.

        Parâmetros:
            contour (np.ndarray): contorno do objeto (como retornado por cv2.findContours).

        Retorna:
            tuple[int, int, int, int] | None:
                (x1, y1, x2, y2) → coordenadas dos cantos do retângulo.
                Retorna None se o contorno for inválido ou vazio.
        """
        # Verifica se o contorno é válido e não vazio
        if contour is None or len(contour) == 0:
            return None

        # Garante que o contorno tenha o formato adequado
        contour = np.array(contour).astype(np.int32)

        # Obtém o retângulo delimitador
        x, y, w, h = cv2.boundingRect(contour)

        # Retorna coordenadas absolutas (x1, y1, x2, y2)
        return x, y, x + w, y + h
    
    def HighlightImg(self, img, dim=25):
        """
        Realça objetos brilhantes em uma imagem em tons de cinza.

        Parâmetros:
            img (np.ndarray): imagem de entrada em escala de cinza.
            dim (int): tamanho do elemento estruturante para o rea00lce (default = 25).

        Retorna:
            np.ndarray: imagem realçada.
        """
        try:
            # Garante que a imagem esteja em escala de cinza
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Cria elemento estruturante elíptico (suave)
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

    #função para reduzir a imagem original
    def ReduceWindow(self, img, coorVetor, d=10):
        """
        Reduz a imagem para uma subjanela baseada nas coordenadas (x, y, w, h)
        aplicando uma leve transformação de perspectiva.

        Parâmetros:
            img (np.ndarray): imagem original.
            coorVetor (list | tuple): [x, y, w, h] delimitando a janela.
            d (int): margem adicional em pixels (default = 10).

        Retorna:
            np.ndarray: imagem reduzida (ou original, em caso de erro).
        """
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

        
    #função responsável para reduzir a imagem para os contornos do campo
    def ReduceField(self, BinImg, Img, fieldWidth, d=10):
        """
        Reduz a imagem para a região do maior contorno (campo).
        """

        # Buscar contornos externos
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


    #função para converter medidas
    def ConvertMeasures(self, w_cm, w_px):
        '''
            Ajusto a constante de proporcionaldiade de px para cm. 
            Representada pela variável: prop_px_cm
        '''
        #atualizando proporções para realizar os devidos cálculos
        self.prop_px_cm = w_px / w_cm
        self.min_diag = (7.5 / 4) * np.sqrt(2) * self.prop_px_cm

    #função para listar os jogadores
    def ListPlayers(self, teamList):
        '''
            Simplesmente lista os jogadores detectados
        '''
        amount = len(teamList)
        for i in range(amount):
            print(teamList[i].team, teamList[i].id)
            print("Posição: x =", teamList[i].pos[0], " y =", teamList[i].pos[1])
        
        print("====================")

    #puxando intervalos de cores
    def CreateColorBounds(self, color_array, hue_tolerance=10,
                            saturation_tolerance=50, value_tolerance=50):
        '''
            Cria as bandas inferior e superior em HSV por meio de um array
            que passa a informação em HSV dada pelo usuário.

            O Hue é tratado com wrap circular (módulo 180, padrão do OpenCV): para
            cores próximas do 0/179 (ex: vermelho) o limite inferior pode ficar MAIOR
            que o superior, indicando um intervalo que cruza o 0°. Consumidores devem
            tratar esse caso:
              - ColorInRange() já lida com o wrap (lower[0] > upper[0]).
              - para cv2.inRange, use self.MaskInRange() (que divide em dois
                intervalos e faz OR), pois cv2.inRange sozinho não suporta wrap.
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
    

    #Desenhar circulos na imagem onde estão os jogadores
    def DrawPlayerCircle(self, imgDegub, robot:Robot):
        '''
            Desenha um círculo no jogador
        '''
        xi = int(robot.xi)
        yi = int(robot.yi)
        ri = int(robot.ri)

        #Configurando prints
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


    def UpSaturation(self, img, boost: int = 50):
        """
        Aumenta a saturação de uma imagem BGR.
        
        Parâmetros:
            img (np.ndarray): imagem em formato BGR.
            boost (int): valor a adicionar à saturação (padrão: +50).
        
        Retorna:
            np.ndarray: imagem com saturação aumentada.
        """
        # Converte para HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Aumenta o canal de saturação (S) de forma vetorizada
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.int16) + boost, 0, 255).astype(np.uint8)

        # Converte de volta para BGR
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    
    def DetectSquares(self,img_bin, min_diag=20):
        """
        Processa uma imagem binarizada e retorna apenas os objetos com formato próximo de quadrado.
        Retorna:
            - bin_res: imagem binarizada com apenas os quadrados;
            - contours_treat: lista de contornos correspondentes aos quadrados.
        """
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

    def FindBinaryContours(self, image, lower_bound, upper_bound):
        '''
        Encontra contornos na imagem capturada filtrando com HSV
        '''
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
    
    #sort values
    def SortPoints(self, points):
        # Ordena os pontos primeiro pelo y (crescente) e depois pelo x (crescente)
        points = sorted(points, key=lambda p: (p[1], p[0]))
        
        # Identificar os dois pontos superiores e os dois pontos inferiores
        top_points = points[:2]
        bottom_points = points[2:]
        
        # Ordenar os pontos superiores por x (crescente)
        top_points = sorted(top_points, key=lambda p: p[0])
        
        # Ordenar os pontos inferiores por x (crescente)
        bottom_points = sorted(bottom_points, key=lambda p: p[0])
        
        # Retornar os pontos na ordem desejada: [top_right, top_left, bottom_left, bottom_right]
        sorted_points = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.int32)
        
        return sorted_points

    # =========== | CALIBRAÇÃO DA CÂMERA | ===========================

    """
    @GNõMIO 2025: Aqui seria interessante, mas seria um bonûs. Pois, só com a homografica
    já garanto resultados suficientemente satisfatórios.
    """
    #=============| Definindo funções módulares | ===========================
    
    def CheckFieldReset(self):
        """
        Verifica se houve muitas falhas consecutivas na detecção do campo
        e executa um reset completo se necessário.
        """
        if self.fieldDetectionFailCount >= self.maxFieldFailures:
            print(f"[VisionSystem] RESET: {self.fieldDetectionFailCount} falhas consecutivas na detecção do campo")
            
            # Reset completo de todos os objetos
            for bot in (*self.allyTeam, *self.enemyTeam):
                bot.reset()  # Reset completo (incluindo Kalman)
                
            if hasattr(self, "ball"):
                self.ball.reset()
                
            # Reset de transformações e configurações
            self.homography_matrix = None
            self.inv_homography_matrix = None
            self.fieldDetectedFlag = False
            
            # Reset do contador (opcional - ou manter para evitar reset contínuo)
            self.fieldDetectionFailCount = 0  # Reset para evitar múltiplos resets
            
            if self.debug:
                print("[FIELD_RESET] Sistema resetado devido a falhas persistentes na detecção do campo")
                

    # ============ GRUPO C - MÉTODOS DE DETECÇÃO SEM FILTRAGEM ADAPTATIVA ============
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

                # Circulando bola no frame de debug
            if self.debug:
                cv2.circle(self.frameResult, (xb, yb), (rb + 2), (0, 0, 255), 2)
                cv2.putText(self.frameResult, "B", (xb, yb - rb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

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
        """
        Detecta robôs na imagem. Pipeline de blob idêntico ao original; a
        diferença só aparece quando uma janela mostra cor de aliado E de
        inimigo em quantidade relevante ao mesmo tempo (colisão) - nesse caso
        o blob é dividido em dois candidatos em vez de descartado ou
        misclassificado.
        """
        # 1. Usa o HSV Global
        if hsv_img is not None:
            imgHSV = hsv_img
        else:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        debug = dbg

        # Cache de últimas posições conhecidas (persiste entre frames).
        if not hasattr(self, "_enemy_last_pos"):
            self._enemy_last_pos = {}   # {slot_idx (0,1,2): (xcm, ycm)}
        if not hasattr(self, "_ally_last_pos"):
            self._ally_last_pos = {}    # {bot_id (ID_Robots): (xcm, ycm)}

        # Reset de contadores e status
        self.playersCount = self.enemiesCount = self.alliesCount = 0
        for bot in (*self.enemyTeam, *self.allyTeam):
            bot.setStatus(False)

        ellipse5 = self.struct_ellipse5
        rect11 = self.struct_rect11

        # --------------------------------------------------------------------
        # Pipeline de blob = ORIGINAL (sem filtro de cor de time aqui).
        # --------------------------------------------------------------------
        self.binaryPlayers = cv2.inRange(imgHSV, self.objectsDarkColor, self.objectsLightColor)

        # Remoção cirúrgica da bola na máscara de objetos
        if self.ball.status:
            xb, yb = int(self.ball.xb), int(self.ball.yb)
            r = 5
            h, w = self.binaryPlayers.shape[:2]
            y1b, y2b = max(0, yb - r), min(h, yb + r)
            x1b, x2b = max(0, xb - r), min(w, xb + r)
            self.binaryPlayers[y1b:y2b, x1b:x2b] = 0

        self.binaryPlayers = cv2.erode(self.binaryPlayers, ellipse5, iterations=1)
        self.binaryPlayers = cv2.morphologyEx(self.binaryPlayers, cv2.MORPH_CLOSE, rect11)
        self.binaryPlayers, players = self.DetectSquares(self.binaryPlayers)

        # Pré-cálculos
        winSize = int(18 * self.prop_px_cm)
        half_win = winSize // 2
        playerRadius = (7.5 / 2) * np.sqrt(2) * self.prop_px_cm
        mainColorRadius = (7.5 / 4) * np.sqrt(5) * self.prop_px_cm

        MAX_JUMP_CM = 100        # gating de distância (FIX #5)
        MIN_COLOR_PIXELS = 8      # ignora ruído mínimo ao checar mistura de cor
        MIX_RATIO_MIN = 0.25      # cor minoritária precisa ter >= 25% da majoritária pra contar como mistura real
        MERGE_SIZE_FACTOR = 1.15  # blob precisa ser >=15% maior que um robô normal pra suspeitar de fusão
        # As quatro constantes acima são só um ponto de partida - vale calibrar
        # olhando imagens reais de colisão do seu time.

        AgoalFlag = Aatk1Flag = Aatk2Flag = False

        if debug:
            binaryAllies = np.zeros(img.shape[:2], dtype=np.uint8)
            binaryAllTeam = np.zeros(img.shape[:2], dtype=np.uint8)

        ally_candidates = []
        enemy_candidates = []

        for currentPlayer in players:
            (xi, yi), ri = cv2.minEnclosingCircle(currentPlayer)

            if debug:
                cv2.circle(self.frameResult, (int(xi), int(yi)), int(ri) + 5, (0, 255, 0), 2)

            if not (0.2 * playerRadius < ri < 2 * playerRadius and self.playersCount < 6):
                continue
            self.playersCount += 1

            x1, y1 = max(0, int(xi - half_win)), max(0, int(yi - half_win))
            x2, y2 = min(img.shape[1], int(xi + half_win)), min(img.shape[0], int(yi + half_win))
            windowActual = img[y1:y2, x1:x2]
            if windowActual.size == 0:
                continue
            hsv = imgHSV[y1:y2, x1:x2]

            # Máscaras de cor (igual ao original)
            mask_ally = self.MaskInRange(hsv, self.ally_lower_bound, self.ally_upper_bound)
            mask_enemy = self.MaskInRange(hsv, self.enemy_lower_bound, self.enemy_upper_bound)
            ally_area = cv2.countNonZero(mask_ally)
            enemy_area = cv2.countNonZero(mask_enemy)
            total_area = max(ally_area + enemy_area, 1)
            ally_ratio = ally_area / total_area
            enemy_ratio = enemy_area / total_area

            is_mixed = (
                ri > MERGE_SIZE_FACTOR * playerRadius
                and ally_area >= MIN_COLOR_PIXELS and enemy_area >= MIN_COLOR_PIXELS
                and min(ally_area, enemy_area) / max(ally_area, enemy_area) >= MIX_RATIO_MIN
            )

            if is_mixed:
                # ------------------------------------------------------------
                # Sinal real de colisão aliado x inimigo: as duas cores estão
                # presentes em quantidade relevante no mesmo blob. Reexamina
                # numa janela um pouco maior (pra caber os dois robôs) e tenta
                # achar, dentro dela, o maior blob de CADA cor separadamente.
                # ------------------------------------------------------------
                margin = int(0.5 * winSize)
                bx1, by1 = max(0, x1 - margin), max(0, y1 - margin)
                bx2, by2 = min(img.shape[1], x2 + margin), min(img.shape[0], y2 + margin)
                hsv_big = imgHSV[by1:by2, bx1:bx2]

                for lower, upper, team_list, is_enemy in (
                    (self.ally_lower_bound, self.ally_upper_bound, ally_candidates, False),
                    (self.enemy_lower_bound, self.enemy_upper_bound, enemy_candidates, True),
                ):
                    m = self.MaskInRange(hsv_big, lower, upper)
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    c = max(cnts, key=cv2.contourArea, default=None)
                    if c is None:
                        continue
                    (lx, ly), lr = cv2.minEnclosingCircle(c)
                    min_rc = (0.75 if is_enemy else 0.5) * mainColorRadius
                    if lr < min_rc:
                        continue

                    gx, gy = lx + bx1, ly + by1  # centro da cor deste robô, em coords globais

                    # Direção aproximada: do centro do blob fundido (xi,yi) até
                    # o centro da cor deste robô - mesma fórmula do caso normal;
                    # sem contorno individual próprio, é a melhor referência
                    # disponível durante a colisão.
                    direction = np.array([xi, -yi]) - np.array([gx, -gy])
                    modDir = np.linalg.norm(direction)
                    if modDir > 1e-6:
                        direction = direction / modDir

                    gxcm, gycm = self.GetPointVirtual(self.TransformPoint(np.array([gx, gy])))

                    wx1, wy1 = max(0, int(gx - half_win)), max(0, int(gy - half_win))
                    wx2, wy2 = min(img.shape[1], int(gx + half_win)), min(img.shape[0], int(gy + half_win))
                    sub_window = img[wy1:wy2, wx1:wx2]
                    if sub_window.size == 0:
                        continue

                    team_list.append({
                        "xi": gx, "yi": gy, "ri": playerRadius,
                        "x_m": gx, "y_m": gy,
                        "xcm": gxcm, "ycm": gycm, "rcm": 5.30,
                        "direction": direction,
                        "windowActual": sub_window,
                        "contour": currentPlayer,
                    })
                continue

            # --------------------------------------------------------------
            # Caso normal (sem mistura) - idêntico ao comportamento original.
            # --------------------------------------------------------------
            if ally_ratio <= 0.4 and enemy_ratio <= 0.4:
                continue  # time indefinido, mesmo comportamento de antes

            team_is_enemy = enemy_ratio > 0.4
            curr_mask = mask_enemy if team_is_enemy else mask_ally

            contour = max(cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                        key=cv2.contourArea, default=None)
            if contour is None:
                continue
            (x_m, y_m), rc = cv2.minEnclosingCircle(contour)
            x_m += x1
            y_m += y1

            min_rc = (0.75 if team_is_enemy else 0.5) * mainColorRadius
            if rc < min_rc:
                continue

            direction = np.array([xi, -yi]) - np.array([x_m, -y_m])
            modDir = np.linalg.norm(direction)
            if modDir > 1e-6:
                direction = direction / modDir

            xcm, ycm = self.GetPointVirtual(self.TransformPoint(np.array([xi, yi])))

            cand = {
                "xi": xi, "yi": yi, "ri": ri,
                "x_m": x_m, "y_m": y_m,
                "xcm": xcm, "ycm": ycm, "rcm": 5.30,
                "direction": direction,
                "windowActual": windowActual,
                "contour": currentPlayer,
            }

            if team_is_enemy:
                enemy_candidates.append(cand)
            else:
                ally_candidates.append(cand)

        # --------------------------------------------------------------------
        # INIMIGOS - pareamento por vizinho mais próximo + gating (FIX #4 e #5)
        # --------------------------------------------------------------------
        if enemy_candidates:
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
            for dist, ci, slot in pairs:
                if ci in used_candidates or slot in used_slots:
                    continue
                last = self._enemy_last_pos.get(slot)
                if last is not None and dist > MAX_JUMP_CM:
                    continue
                assignment[slot] = ci
                used_candidates.add(ci)
                used_slots.add(slot)

            for slot, ci in assignment.items():
                cand = enemy_candidates[ci]
                bot = self.enemyTeam[slot]

                Color_p_pt, Color_s_pt, _ = self.GetCentersColors(cand["xi"], cand["yi"], cand["x_m"], cand["y_m"])
                Color_p = self.GetHsvMean(imgHSV, Color_p_pt[0], Color_p_pt[1])
                Color_s = self.GetHsvMean(imgHSV, Color_s_pt[0], Color_s_pt[1])

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

                if debug:
                    self.DrawPlayerCircle(self.frameResult, bot)
                    cx, cy = int(Color_p_pt[0]), int(Color_p_pt[1])
                    bgr_p = self.Hsv2Bgr(Color_p)
                    cv2.circle(self.frameResult, (cx, cy), 4, (0, 0, 0), -1)
                    cv2.circle(self.frameResult, (cx, cy), 3, bgr_p, -1)
                    cx2, cy2 = int(Color_s_pt[0]), int(Color_s_pt[1])
                    bgr_s = self.Hsv2Bgr(Color_s)
                    cv2.circle(self.frameResult, (cx2, cy2), 4, (0, 0, 0), -1)
                    cv2.circle(self.frameResult, (cx2, cy2), 3, bgr_s, -1)

        # --------------------------------------------------------------------
        # ALIADOS - identidade por cor secundária conhecida + gating (FIX #5)
        # --------------------------------------------------------------------
        for cand in ally_candidates:
            if self.alliesCount >= 3:
                break

            ally_checks = [
                (not AgoalFlag, self.goalAllyColor1, self.goalAllyColor2, ID_Robots.ROBOT_ALLY_GOAL, "Goleiro"),
                (not Aatk1Flag, self.atk1AllyColor1, self.atk1AllyColor2, ID_Robots.ROBOT_ALLY_1, "Atacante 1"),
                (not Aatk2Flag, self.atk2AllyColor1, self.atk2AllyColor2, ID_Robots.ROBOT_ALLY_2, "Atacante 2"),
            ]

            assigned = False
            for flag, c1, c2, bot_id, name in ally_checks:
                if not (flag and self.DetectAllyRobot(cand["windowActual"], c1, c2)):
                    continue

                last = self._ally_last_pos.get(bot_id)
                if last is not None:
                    dist = float(np.hypot(cand["xcm"] - last[0], cand["ycm"] - last[1]))
                    if dist > MAX_JUMP_CM:
                        # Provável contaminação de cor por um robô vizinho -
                        # ignora esse match e tenta o próximo ally_check.
                        continue

                bot = self.allyTeam[bot_id]

                if not bot.kalman_initialized:
                    bot.setPosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)
                else:
                    bot.updatePosition(cand["xcm"], cand["ycm"], cand["direction"], cand["windowActual"], time=timestamp)

                bot.updtPositionImg(cand["xi"], cand["yi"], cand["ri"])
                bot.setStatus(True)
                bot.setRadius(cand["rcm"])
                bot.setColor(colorT=self.allyColor, colorP=c1, colorS=c2)
                if debug:
                    self.DrawPlayerCircle(self.frameResult, bot)
                self.DrawPlayerVirtual(bot)

                self._ally_last_pos[bot_id] = (cand["xcm"], cand["ycm"])

                if bot_id == ID_Robots.ROBOT_ALLY_GOAL:
                    AgoalFlag = True
                elif bot_id == ID_Robots.ROBOT_ALLY_1:
                    Aatk1Flag = True
                else:
                    Aatk2Flag = True

                if debug:
                    cv2.circle(self.frameResult, (int(cand["x_m"]), int(cand["y_m"])), 4, (255, 128, 255), -1)

                assigned = True
                break

            if not assigned:
                continue

            self.alliesCount = min(self.alliesCount + 1, 3)
            if debug:
                cv2.drawContours(binaryAllies, [cand["contour"]], -1, 255, -1)
                cv2.drawContours(binaryAllTeam, [cand["contour"]], -1, 255, -1)

        if debug:
            self.binaryAllies = binaryAllies
            self.binaryAllTeam = binaryAllTeam
        self._countProcess += 1

    #desenhar robôs na imagem
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


#==========================| MÉTODOS DE PREVISÃO DO KALMAN | ================================================
    #prevendo posição da bola
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


# ==========================| Utilidades para métodos estruturados | =============================================
    # Método novo para procurar se existe um robô na janela
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
        '''
            Converte HSV para BGR
        '''
        hsv_pixel = np.uint8([[color_hsv]])   # shape (1,1,3)
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in bgr_pixel[0,0])

    def ColorInRange(self, hsv_values, lower, upper):
        """
        Verifica se uma ou várias cores HSV estão dentro da faixa especificada.
        Suporta faixas que cruzam o limite do Hue (ex: vermelho 170–10).

        Parâmetros:
            hsv_values : np.ndarray
                Cor única [H, S, V] ou matriz Nx3 com várias cores HSV.
            lower : iterable
                Limite inferior [H, S, V].
            upper : iterable
                Limite superior [H, S, V].

        Retorna:
            np.ndarray (bool) se hsv_values for Nx3, ou bool se for [3]
        """
        hsv_values = np.atleast_2d(hsv_values).astype(np.uint16)
        lower = np.array(lower, dtype=np.uint16)
        upper = np.array(upper, dtype=np.uint16)

        # --- caso em que faixa cruza o 0° do Hue (ex: 170–10)
        if lower[0] > upper[0]:
            mask_hue = ((hsv_values[:, 0] >= lower[0]) | (hsv_values[:, 0] <= upper[0]))
        else:
            mask_hue = ((hsv_values[:, 0] >= lower[0]) & (hsv_values[:, 0] <= upper[0]))

        # --- máscaras para S e V (sempre diretas)
        mask_sat = ((hsv_values[:, 1] >= lower[1]) & (hsv_values[:, 1] <= upper[1]))
        mask_val = ((hsv_values[:, 2] >= lower[2]) & (hsv_values[:, 2] <= upper[2]))

        # Combina tudo
        mask = mask_hue & mask_sat & mask_val

        # Retorna booleano simples se foi entrada única
        return mask[0] if hsv_values.shape[0] == 1 else mask

    def IsColorMatch(self, measured_hsv, target_hsv):
        lower, upper = self.CreateColorBounds(target_hsv)
        return self.ColorInRange(measured_hsv, lower, upper)

    # ================= Método simplificado de processamento ==========
    # ============ GRUPO D - FUNÇÕES PARA DETECÇÃO ADAPTATIVA ============
    def SearchBots(self, img, timestamp, debug=False) -> list:
            if img is None:
                return []

            H, W = img.shape[:2]
            
            detected_list = []

            # Lista de alvos: (ObjetoRobo, EnumTime)
            targets = []
            for b in self.allyTeam:
                targets.append((b, ID_Team.TEAM_ALLY))
            for b in self.enemyTeam:
                targets.append((b, ID_Team.TEAM_ENEMY))

            for bot, team_enum in targets:
                # 1. PREDIÇÃO: Pega imagem cortada e retângulo
                roi_img, roi_rect = self.PredictRobot([H, W], team_enum, bot.id, timestamp)

                # Se não retornou imagem válida (ex: fora do campo), pula
                if roi_img is None or roi_img.size == 0:
                    continue

                # 2. DETECÇÃO: Passa a imagem cortada
                # Nota: 'img' global não é passada, passamos 'roi_img'
                result = self.DetectBotInRoi(roi_img, roi_rect, bot, debug)

                if result:
                    detected_list.extend(result)
                    
                    # Debug Visual: Desenhar o retângulo onde o robô foi buscado
                    if debug:
                        xr, yr, wr, hr = roi_rect
                        cv2.rectangle(self.frameResult, (xr, yr), (xr+wr, yr+hr), (0, 255, 0), 1)

            return detected_list


    def DetectBotInRoi(self, roi_img, roi_rect, target_bot, debug=False) -> list:
            """
            Processa o ROI para achar candidatos e recorta uma janela menor (bot_win)
            para análise de cor e direção, retornando-a nos resultados.
            """
            # 0. Validações básicas
            if roi_img is None or roi_img.size == 0:
                return []

            # Desempacota offsets globais (x0, y0 é o canto sup. esq. do ROI no campo)
            x0, y0, w0, h0 = roi_rect
            results = []

            # --------------------------------------------------------
            # 1. Pré-processamento no ROI (igual ao antigo search_bot, mas no ROI)
            # --------------------------------------------------------
            # Usamos as coordenadas do roi_rect para cortar a matriz global
            roi_hsv = self.imgHSV[y0 : y0 + h0, x0 : x0 + w0]

            obj_mask = cv2.inRange(roi_hsv, self.objectsDarkColor, self.objectsLightColor)

            # Remover Bola e Jogadores (Subtração das máscaras globais recortadas)
            if self.binaryBall is not None:
                ball_roi = self.binaryBall[y0:y0+h0, x0:x0+w0] # Slice global
                if ball_roi.shape == obj_mask.shape:
                    obj_mask = cv2.subtract(obj_mask, ball_roi)

            if self.binaryPlayers is not None:
                p_roi = self.binaryPlayers[y0:y0+h0, x0:x0+w0] # Slice global
                if p_roi.shape == obj_mask.shape:
                    obj_mask = cv2.subtract(obj_mask, p_roi)

            # Morfologia
            mask = cv2.erode(obj_mask, self.struct_ellipse5, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.struct_rect11)

            # Detecção de contornos (Candidatos)
            _, candidates = self.DetectSquares(mask)
            if not candidates:
                return []

            # Parâmetros de tamanho (Convertidos de cm para px)
            winSize = int(18 * self.prop_px_cm)
            half_win = winSize // 2
            playerRadius = (7.5 / 2) * np.sqrt(2) * self.prop_px_cm
            mainColorRadius = (7.5 / 4) * np.sqrt(5) * self.prop_px_cm*1.3

            # --------------------------------------------------------
            # 2. Loop pelos Candidatos
            # --------------------------------------------------------
            for cnt in candidates:
                # Centro e Raio LOCAL (relativo ao roi_img)
                (xi_local, yi_local), ri = cv2.minEnclosingCircle(cnt)

                # Cálculo das coordenadas GLOBAIS
                xi_global = int(xi_local + x0)
                yi_global = int(yi_local + y0)

                # Filtro de Tamanho (igual ao original)
                if not (0.2 * playerRadius < ri < 2 * playerRadius):
                    # if debug: print("Ignorado por tamanho")
                    continue

                # --------------------------------------------------------
                # 3. Recorte do bot_win (Janela do Robô)
                # --------------------------------------------------------
                # No código antigo, você cortava da imagem 'img' global usando xi_global.
                # Aqui, cortamos de 'roi_img' usando 'xi_local'.
                
                # Limites dentro do roi_img
                x1_local = max(0, int(xi_local - half_win))
                y1_local = max(0, int(yi_local - half_win))
                x2_local = min(w0, int(xi_local + half_win))
                y2_local = min(h0, int(yi_local + half_win))

                # A 'bot_win' é o recorte específico do robô
                bot_win = roi_img[y1_local:y2_local, x1_local:x2_local]
                
                if bot_win.size == 0: continue

                # HSV dessa pequena janela
                hsv_win = roi_hsv[y1_local:y2_local, x1_local:x2_local]

                # --------------------------------------------------------
                # 4. Análise de Cores (Time) dentro de bot_win
                # --------------------------------------------------------
                mask_ally = self.MaskInRange(hsv_win, self.ally_lower_bound, self.ally_upper_bound)
                mask_enemy = self.MaskInRange(hsv_win, self.enemy_lower_bound, self.enemy_upper_bound)

                ally_area = cv2.countNonZero(mask_ally)
                enemy_area = cv2.countNonZero(mask_enemy)
                total_area = max(ally_area + enemy_area, 1)

                detected_team = None
                curr_mask = None
                curr_main_color = None

                if (ally_area / total_area) > 0.55:
                    detected_team = ID_Team.TEAM_ALLY
                    curr_mask = mask_ally
                    curr_main_color = self.allyColor
                elif (enemy_area / total_area) > 0.55:
                    detected_team = ID_Team.TEAM_ENEMY
                    curr_mask = mask_enemy
                    curr_main_color = self.enemyColor
                else:
                    continue # Time indefinido

                # --------------------------------------------------------
                # 5. Determinar Direção (Centro da Cor)
                # --------------------------------------------------------
                sub_cnts, _ = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                contour_color = max(sub_cnts, key=cv2.contourArea, default=None)
                if contour_color is None:
                    continue

                (xm_win, ym_win), rc = cv2.minEnclosingCircle(contour_color)

                # Coordenadas GLOBAIS da mancha de cor
                # xm_win é relativo a bot_win -> somar x1_local -> relativo a roi_img -> somar x0 -> Global
                xm_global = xm_win + x1_local + x0
                ym_global = ym_win + y1_local + y0

                if rc < 0.5 * mainColorRadius:
                    continue

                # Cálculo da Direção: (Centro Robô - Centro Cor)
                # Igual ao seu código original: direction = np.array([xi_global, -yi_global]) - np.array([x_m, -y_m])
                dx = xi_global - xm_global
                dy = -(yi_global - ym_global) # Inverte Y conforme seu padrão

                direction = np.array([dx, dy], dtype=float)
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction /= norm
                else:
                    direction = np.array([1.0, 0.0])

                theta = float(np.arctan2(direction[1], direction[0]))

                # --------------------------------------------------------
                # 6. Identificação do ID
                # --------------------------------------------------------
                # Extrair cores nos pontos geométricos globais
                C_p, C_s, _ = self.GetCentersColors(xi_global, yi_global, xm_global, ym_global)
                
                # Usa imgHSV global (self.imgHSV deve estar atualizado no filtered_detection)
                primary = self.GetHsvMean(self.imgHSV, C_p[0], C_p[1])
                secondary = self.GetHsvMean(self.imgHSV, C_s[0], C_s[1])

                identified_id = None

                # a) Verifica match com o alvo
                if detected_team == target_bot.team:
                    if self.MatchesRobot(target_bot, primary, secondary):
                        identified_id = target_bot.id

                # b) Verifica match com outros (Arvore de cores)
                if identified_id is None:
                    match = self.colorTree.find_by_colors(curr_main_color, primary, secondary)
                    if match:
                        cand_id = match["robot_id"]
                        # Valida se existe no time
                        team_list = self.allyTeam if detected_team == ID_Team.TEAM_ALLY else self.enemyTeam
                        if any(b.id == cand_id for b in team_list):
                            identified_id = cand_id

                # --------------------------------------------------------
                # 7. Compilação dos Resultados
                # --------------------------------------------------------
                if identified_id is not None:
                    # Conversão px -> cm
                    xv, yv = self.TransformPoint(np.array([xi_global, yi_global]))
                    xcm, ycm = self.GetPointVirtual(np.array([xv, yv]))

                    result_data = {
                        "id": identified_id,
                        "team": detected_team,
                        "x": xcm,
                        "y": ycm,
                        "theta": theta,
                        "direction": direction,
                        "img_x": xi_global,
                        "img_y": yi_global,
                        "img_r": ri,
                        "xi_image": int(xm_global), # Centro da cor (debug)
                        "yi_image": int(ym_global),
                        "bot_win": bot_win, # <--- AQUI ESTÁ A WINDOW RETORNADA
                        "contour_global": None 
                    }

                    # Prepara contorno global para desenhar na máscara global depois
                    c_global = cnt.copy()
                    c_global[:, 0, 0] += x0
                    c_global[:, 0, 1] += y0
                    result_data["contour_global"] = c_global

                    # Atualiza máscara global (opcional, igual ao original)
                    if self.binaryPlayers is not None:
                        cv2.drawContours(self.binaryPlayers, [c_global], -1, 255, -1)
                    
                    results.append(result_data)

            return results

    #=================================================================================

    def GetBotById(self, team:ID_Team, bot_id:ID_Robots) -> Robot:
        '''
            retorna o robô por meio do identificador e do time.
        '''
        if team == ID_Team.TEAM_ALLY:
            return self.allyTeam[bot_id]
        else:
            return self.enemyTeam[bot_id]
            
    def SearchBot(self, img, roi, team: ID_Team, bot_id: ID_Robots, timestamp, debug=False):
        """
        Procura um robô específico na ROI, mas atualiza também outros robôs do mesmo time
        se forem detectados.
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
        # Preparar janela e máscara de objetos
        # --------------------------------------------------------
        # Recorte direto da ROI
        window = img[y0:y0+h0, x0:x0+w0]
        windowHSV = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
        obj_window = cv2.inRange(windowHSV, self.objectsDarkColor, self.objectsLightColor)

        if self.binaryBall is not None:
            obj_window = cv2.subtract(obj_window, self.binaryBall[y0:y0+h0, x0:x0+w0])
        if self.binaryPlayers is not None:
            obj_window = cv2.subtract(obj_window, self.binaryPlayers[y0:y0+h0, x0:x0+w0])

        # --------------------------------------------------------
        # Morfologia e detecção de candidatos
        # --------------------------------------------------------
        ellipse5 = self.struct_ellipse5
        rect11   = self.struct_rect11
        binaryPlayer = cv2.erode(obj_window, ellipse5, iterations=1)
        binaryPlayer = cv2.morphologyEx(binaryPlayer, cv2.MORPH_CLOSE, rect11)
        _, candidates = self.DetectSquares(binaryPlayer)

        if not candidates:
            if debug:
                print(f"[search_bot] Nenhum candidato detectado na ROI")
            return False

        winSize = int(18 * self.prop_px_cm)
        half_win = winSize // 2
        playerRadius = (7.5 / 2) * np.sqrt(2) * self.prop_px_cm
        mainColorRadius = (7.5 / 4) * np.sqrt(5) * self.prop_px_cm
        found = False

        # --------------------------------------------------------
        # Processamento dos candidatos
        # --------------------------------------------------------
        for contour in candidates:
            (xi, yi), ri = cv2.minEnclosingCircle(contour)
            xi_global = xi + x0
            yi_global = yi + y0

            if not (0.2 * playerRadius < ri < 2 * playerRadius and self.playersCount < 6):
                if debug:
                    print("  ⚠️ Ignorado (fora do range esperado ou excedeu limite).")
                continue

            x1 = max(0, int(xi_global - half_win))
            y1 = max(0, int(yi_global - half_win))
            x2 = min(self.fieldReduce.shape[1], int(xi_global + half_win))
            y2 = min(self.fieldReduce.shape[0], int(yi_global + half_win))

            # Janela menor ainda para processamento do robô
            bot_win = img[y1:y2, x1:x2]
            hsv_win = cv2.cvtColor(bot_win, cv2.COLOR_BGR2HSV)

            # Máscaras de cor
            mask_ally = self.MaskInRange(hsv_win, self.ally_lower_bound, self.ally_upper_bound)
            mask_enemy = self.MaskInRange(hsv_win, self.enemy_lower_bound, self.enemy_upper_bound)

            ally_area = cv2.countNonZero(mask_ally)
            enemy_area = cv2.countNonZero(mask_enemy)
            total_area = max(ally_area + enemy_area, 1)

            #Proporção das áreas
            ally_ratio = ally_area / total_area
            enemy_ratio = enemy_area / total_area

            if ally_ratio > 0.55:
                team_type = "ally"
                mainColor = self.allyColor
                teamBots = self.allyTeam
            elif enemy_ratio > 0.55:
                team_type = "enemy"
                mainColor = self.enemyColor
                teamBots = self.enemyTeam
            else:
                continue 

            #Coordenadas absolutas:
            xcm, ycm = self.GetPointVirtual(self.TransformPoint(np.array([xi, yi])))

            # ----------------------------------------------------
            # Determinar coordenadas globais do centro da cor principal
            # ----------------------------------------------------
            contour_color = max(cv2.findContours(mask_enemy if team_type=="enemy" else mask_ally,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)[0],
                                key=cv2.contourArea, default=None)
            if contour_color is None:
                continue

            (x_m, y_m), rc = cv2.minEnclosingCircle(contour_color)
            # Corrigindo para coordenadas gerais
            # o indice está numa janela dentro de outra janela
            # então a coordenada real é somando o extremo roi do roi
            x_m += x1
            y_m += y1

            if rc < 0.6 * mainColorRadius: #raio da cor maior tem que ser considerável
                continue

            direction = np.array([xi_global, -yi_global]) - np.array([x_m, -y_m])
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm

            # ----------------------------------------------------
            # Extrair cores e tentar identificar robô
            # ----------------------------------------------------
            C_p, C_s, _ = self.GetCentersColors(xi_global, yi_global, x_m, y_m)
            primary = self.GetHsvMean(self.imgHSV, C_p[0], C_p[1])
            secondary = self.GetHsvMean(self.imgHSV, C_s[0], C_s[1])

            # Tenta o robô específico
            if self.MatchesRobot(bot, primary, secondary):
                # O robô corresponde, então só atualiza a posição.
                bot.updatePosition(x=xcm, y=ycm, direction=direction, image=bot_win, time=timestamp)
                bot.updtPositionImg(xi_global, yi_global, ri)
                bot.setStatus(True)
                self.DrawPlayerCircle(self.frameResult, bot)
                self.DrawPlayerVirtual(bot)
                found = True
                if debug:
                    print(f"✅ Detectado Robô {team.name} {bot_id.name} (específico) na ROI {roi}")

            else:
                # Verifico se ele corresponde a algum robô do mesmo time
                match = self.colorTree.find_by_colors(mainColor, primary, secondary)
                if match is None:
                    continue 

                bot_id = match['robot_id']
                other_bot = teamBots[bot_id]
                other_bot.updatePosition(x=xcm, y=ycm, direction=direction, image=bot_win, time=timestamp)
                other_bot.updtPositionImg(xi_global, yi_global, ri)
                other_bot.setStatus(True)
                self.DrawPlayerCircle(self.frameResult, other_bot)
                self.DrawPlayerVirtual(other_bot)
                if debug:
                    print(f"✅ Detectado Robô {team.name} {bot_id.name} (específico) na ROI {roi}")


            # Atualiza binário global
            cont_global = contour.copy()
            cont_global[:, 0, 0] += x0
            cont_global[:, 0, 1] += y0
            if self.binaryPlayers is not None:
                cv2.drawContours(self.binaryPlayers, [cont_global], -1, 255, -1)
            if team_type == "ally" and self.binaryAllTeam is not None:
                cv2.drawContours(self.binaryAllTeam, [cont_global], -1, 255, -1)

        return found
    
    def MatchesRobot(self, bot: Robot, primary, secondary) -> bool:
        """
        Verifica se a combinação de cores detectada corresponde ao robô esperado.
        """
        team = bot.team
        robot_id = bot.id
        main_color = bot.colorTeam

        match = self.colorTree.find_by_colors(main_color, primary, secondary)
        if match is None:
            return False

        return match['robot_id'] == robot_id and match['team'] == team


    #=================================================================================
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
            # signature: setPositionNoKalman(x, y, theta, timestamp, image=None)
            bot.setPositionNoKalman(x_pred, y_pred, theta_pred, timestamp, image=None)

            # status: por padrão fallback não é "detectado" (mas escolha sua política)
            bot.detected = bool(mark_detected)

            # update bbox/view já feito por setPositionNoKalman
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
        Usa ball.predict/predict_with_cov e atualiza com setPositionNoKalman da Ball.
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
            # usar API da bola que você definiu: setPositionNoKalman(x, y, r, timestamp=0.0, theta=None)
            rb = getattr(self, "ballRadiusP", self.ball.radius)
            self.ball.setPositionNoKalman(x_pred, y_pred, rb, timestamp, theta=theta_pred)

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
                bot.setPositionNoKalman(pred_x, pred_y, pred_theta, timestamp)
                bot.setStatus(True) # Mantém visualmente ativo
            else:
                # Perda Real: Desliga e marca para reset
                bot.setStatus(False)
                self.robot_kalman_reset_flags[key] = True


    def FilteredDetection(self, img, currentTime, debug=False):
        """
        Pipeline otimizado:
        1. Validações e Recorte do Campo.
        2. Predição e Atualização da Bola.
        3. Predição e Atualização dos Robôs (Delegando lógica interna para a classe Robot).
        """
        print("DEBUG!!! Está com filtered detection!!!")
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
        self.binaryAllTeam = np.zeros((H, W), dtype=np.uint8)
        self.binaryObjects = np.zeros((H, W), dtype=np.uint8)
        self.binaryAllies = np.zeros((H, W), dtype=np.uint8)
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

                # --- ATUALIZAÇÃO CENTRALIZADA NO ROBÔ ---
                # O robô cuida do Kalman, direção, bbox e guarda a imagem
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
        for bot, team in [(b, ID_Team.TEAM_ALLY) for b in self.allyTeam] + \
                        [(b, ID_Team.TEAM_ENEMY) for b in self.enemyTeam]:
            key = (bot.id, team)
            if key not in updated_keys:
                self.SafeCall(self._handle_robot_loss, bot, team, currentTime)

        return self.frameResult
    
    # Adicione este método dentro da classe VisionSystem
    def GetShape(self, img):
        """Retorna (altura, largura) independente se é CPU (numpy) ou GPU (UMat)."""
        if isinstance(img, cv2.UMat):
            return img.get().shape[:2]
        return img.shape[:2]
    
    #=================================================================================
# Testar função principal e nova lógica
if __name__ =='__main__':
    print("Utilizada em função de main")