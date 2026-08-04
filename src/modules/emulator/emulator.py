'''
    @GNOMIO: Motor principal do sistema VSSS. Gerencia o pipeline de processamento,
    coordena threads de visão e comunicação, e mantém o estado global do sistema.

    Versão: v3.0.1
    Última modificação: 15/11/2025
    Autor: Saulo (update)

    Patch Notes v3.0.1:
    - Implementado novo sistema de threads com melhor desempenho
    - Separação de processamento em visão e comunicação
    - Melhor gerenciamento de recursos
    - Sistema de filas otimizado para UI e comunicação
    - Novo sistema de debug com menor overhead
    - Correções na estabilidade do processamento
'''
#=============================================================
from modules.VisionSys.detectorV2 import VisionSystem
from imports import *
from ui.settingsMenu import *
from ui.viewer import MyViewer, WindowsViewer
from ui.cards import *

from modules.VisionSys.components.objects import *
from modules.communication.communication import *
from modules.communication.ui.interface import *

from modules.control.comm.public.state_tx import StatesTransmissor
from modules.control.comm.public.commands_rx import CommandsReceiver
from modules.control.comm.protocols import command_pb2 # Importante para tipagem se necessário

import threading
import queue
import time
from collections import deque
import ast

import traceback

#=============================================================
class Emulator:
    '''
        Emulador é o nome dado à Engine que controla o sistema de captura, controle, comunicação e visão computacional.
    '''
    def __init__(self, App):
        print("[EMULATOR] Construindo instância do emulador...")
        self._init_app_references(App)
        self._init_state_variables()
        self._init_collections()
        self._init_parallel_processing()
        self._init_viewers()
        self._init_control_system()
        self._init_timers(App)
        self._init_system_info(App)
        self._init_gpu_info()
        self._init_camera_settings()
        self._init_capture_and_vision()

        print("[EMULATOR] Inicialização concluída com sucesso.")

    # ==============================================================
    #  1. Referências principais da aplicação (UI, botões, viewers)
    # ==============================================================
    def _init_app_references(self, App):
        """Carrega referências vindas da aplicação principal."""
        self.settingsTree = App.menu
        self.viewer = App.viewer
        self.debugFieldViewer = App.debugField
        self.debugObjectsViewer = App.debugObject
        self.debugPlayersViewer = App.debugPlayers
        self.debugTeamViewer = App.debugTeam
        self.resultViewer = App.result
        self.virtualResult = App.virtualVision

        self.cards = App.cards
        self.infoCards = App.infosEmulator

        self.IdCap = App.IdFrame
        self.btn_run = App.btn_run
        self.btn_pause = App.btn_pause
        self.btn_stop = App.btn_stop

        # Salvando instância da aplicação original
        self.app = App

    # ==============================================================
    #  2. Variáveis de estado e controle
    # ==============================================================
    def _init_state_variables(self):
        """Inicializa variáveis de controle e estado interno do emulador."""
        self.Mode = MODE_DEFAULT
        self.cameraIsRunning = False
        self.DEBUGA = False

        # Para processamento do vídeo
        self.video_active = False      # controla se o loop de vídeo está ativo
        self.video_paused = False      # indica pausa
        self.video_thread = None       # thread dedicada ao vídeo

        self.thread = None
        self.delay = 17  # ms

        self.clientMQTT = None
        self.clientSerial = None
        self.commands = None

        # Variável interna para controle da comunicação de control
        self.comm_running = False 

        self.captureThread = None
        self.frame = None
        self.errorCode = 0
        self.tproc_string = ""

        # Classe de controle da comunicação
        self.comm = None

        # Janela de comunicação
        self.comm_debug_window = None
        self.should_open_comm_window = False

        self.hasCuda = False
        self.CudaDevice = None
        self.CudaDeviceVersion = None
        self.CUDAselected = False

        # Tempo de comunicação
        self.comm_send_interval = 0.100 #10 FPS - Ajustado para evitar sobrecarga na serial 115200 bps

    # ==============================================================
    #  2.1 Processamento paralelo
    #= =============================================================
    def _init_parallel_processing(self):
        """Configura variáveis para controle de processamento paralelo."""
        # Fila original mantida para compatibilidade, mas não será mais usada diretamente
        self.output_queue = queue.Queue(maxsize=1)

        # Filas separadas — independentes para UI e comunicação
        self.ui_queue = queue.Queue(maxsize=1)
        self.comm_queue = queue.Queue(maxsize=1)

        # filas para controle de envio do UDP
        self.udp_send_queue = queue.Queue(maxsize=1)    # Visão -> UDP
        self.udp_recv_queue = queue.Queue(maxsize=1)    # UDP -> Comunicação

        # Threads
        self.vision_thread = None
        self.comm_thread = None

        # Thread do monitoramento da UDP
        self.udp_thread = None 


    # ==============================================================
    # Processamento paralelo: thread de visão e loop da UI, comunicação e controle
    # ==============================================================
    # Thread de captura e processamento das imagens (modo câmera)
    def visionThread(self):
        """Thread separada que faz captura e processamento de visão (modo câmera)."""
        print("[VISION THREAD] Iniciada.")
        while self.cameraIsRunning:
            loop_start = time.time()

            try:
                # --- Captura ---
                frame = self.capture.getImage() #A câmera tem um FPS de 30, então fica travado a 30 FPS o sistema.
                if frame is None:
                    # Meu FPS é limitado pela velocidade de aquisição de dados da câmera.
                    time.sleep(0.001)
                    continue

                # --- Processamento de visão ---
                t_proc_start = time.time()
                result = self.vs.ProcessImg(frame, self.DEBUGA)
                objects = self.vs.GetObjects()
                virtual = self.vs.virtualImg
                t_proc_end = time.time()

                # --- Atualiza tempos ---
                self.frameTime = (t_proc_end - t_proc_start) * 1000.0  # processamento
                self.totalTime = (time.time() - loop_start) * 1000.0   # visão total
                self.realTime = self.Timer.getElapsedTime()/1000.0           # desde init
                #print("[EMULADOR]: Tempo total em segundos ", self.realTime)
          
                self.fill_deques_time()

                # Atualizar string de tempos de processamento
                campo = self.vs.bmk.get_avg("Campo")
                bola = self.vs.bmk.get_avg("Bola")
                players = self.vs.bmk.get_avg("Players")
                self.tproc_string = f"{campo:.2f} / {bola:.2f} / {players:.2f}"

                # --- Atualiza objetos detectados ---
                self.field = objects.get(ID_Objects.FIELD, self.field)
                self.ball = objects.get(ID_Objects.BALL, self.ball)
                self.allies = objects.get(ID_Objects.ALLIES, self.allies)
                self.enemies = objects.get(ID_Objects.ENEMIES, self.enemies)

                # --- Prepara pacote para a UI ---
                data = {
                    'frame': frame,
                    'result': result,
                    'virtual': virtual,
                    'field': self.field,
                    'ball': self.ball,
                    'allies': self.allies,
                    'enemies': self.enemies,
                    'vision_time': self.totalTime,
                    'proc_time': self.frameTime,
                    'real_time': self.realTime,
                    'fps': self.FPStime,
                    'timestamp': self.realTime
                }

                # 1 Gera o frame no formato Protobuf usando seu novo método
                pb_frame = self.vs.GetFrameProtobuff()
                                

                # 2. Envia para a Thread UDP (sem bloquear a visão)
                try:
                    self.udp_send_queue.put(pb_frame, block=False)
                except queue.Full:
                    pass # Se a fila encheu, descartamos o frame antigo, priorizando o atual
                        
                # --- Envia para fila da UI ---
                if self.ui_queue.full():
                    try:
                        self.ui_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.ui_queue.put_nowait(data)

                # --- Envia para fila da comunicação ---
                if self.comm_queue.full():
                    try:
                        self.comm_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.comm_queue.put_nowait(data)

            except Exception as e:
                print("[VISION THREAD] Erro:", e)
                traceback.print_exc()
                time.sleep(0.001)
                continue

        print("[VISION THREAD] Finalizada.")

    # Apenas para atualizar a UI (chamada pela thread principal via .after)
    def updateUI(self):
        """Atualiza a interface com base nas informações das threads (câmera ou vídeo)."""
        # Se não houver processo ativo (nem câmera, nem vídeo), para o loop
        if not self.cameraIsRunning and not self.video_active:
            self.Timer.stop()
            return

        try:
            data = self.ui_queue.get_nowait()
            frame = data['frame']
            result = data['result']
            virtual = data['virtual']
            self.field = data['field']
            self.ball = data['ball']
            self.allies = data['allies']
            self.enemies = data['enemies']
            self.totalTime = data['vision_time']
            self.frameTime = data['proc_time']
            self.realTime = data['real_time']
            self.FPStime = data['fps']

            # --- Exibição segura ---
            if frame is not None:
                self.viewer.show(frame)
            if result is not None:
                # Garantir que result seja exibível
                try:
                    self.resultViewer.show(result)
                except Exception as e:
                    print(f"[UPDATE UI] Erro ao exibir result: {e}")
            else:
                # Se result for None, limpar a aba
                self.resultViewer.clear()

            if virtual is not None:
                self.virtualResult.show(virtual)

            # ============================================================
            # CORREÇÃO: Exibição de todas as imagens de debug
            # ============================================================
            if self.DEBUGA:
                try:
                    imgs = self.vs.GetDebugImages()
                    if imgs and len(imgs) >= 4:
                        self.debugFieldViewer.show(imgs[0])      # Debug campo
                        self.debugObjectsViewer.show(imgs[1])    # Debug bola
                        self.debugPlayersViewer.show(imgs[2])    # Debug jogadores
                        self.debugTeamViewer.show(imgs[3])       # Debug time
                except Exception:
                    pass
            else:
                # Limpa todas as abas de debug quando desativado
                self.debugFieldViewer.clear()
                self.debugObjectsViewer.clear()
                self.debugPlayersViewer.clear()
                self.debugTeamViewer.clear()

            self.setContentRobots()
            
            # --- Atualiza UI (info cards e controle) ---
            avg_proc = self.avg(self.deque_proc)
            self.FPStime = int(1000/avg_proc) if avg_proc > 0 else 0

            self.infoCards.updateInfo("FPS:", self.FPStime)
            self.infoCards.updateInfo("Vision. (ms):", f"{avg_proc:.2f}")
            self.infoCards.updateInfo("Proc. (ms):", f"{self.avg(self.deque_proc):.2f}")
            self.infoCards.updateInfo("Envio (ms):", f"{self.avg(self.deque_send):.2f}")
            self.infoCards.updateInfo("Timer (s):", f"{self.realTime / 1000:.2f}")
            self.infoCards.updateInfo("(F/B/P).(ms):", self.tproc_string)   # ← descomentado

            self.infoCards.update()

        except queue.Empty:
            pass

        # Loop contínuo (~60 FPS) enquanto houver processo ativo
        if self.cameraIsRunning or self.video_active:
            self.viewer.window.after(self.delay, self.updateUI)
            
    # ========================================================================================================
    def InitProtobuff(self):
        """
        Método responsável por inicializar a comunicação do protobuff via UDP que é utilizado pelo modo de controle 
        do sistema.
        """
        pass 
    # ========================================================================================================
    def strategyUDPThread(self):
        # Configura IPs (Defina self.ip_send/recv no __init__ ou use hardcoded para teste)
        self.tx = StatesTransmissor(ip=self.ip_send, port=self.port_send)
        self.rx = CommandsReceiver(ip=self.ip_receive, port=self.port_receive)
        self.rx.receiver_socket.setblocking(False) 

        print("[UDP THREAD] Iniciada.")
        while self.comm_running:
            # 1) --- ENVIAR ESTADO (StateTransmitter) ---
            try:
                frame_to_send = self.udp_send_queue.get(timeout=0.005) # Timeout curto
                self.tx.transmit(frame_to_send)
            except queue.Empty:
                pass

            # 2) --- RECEBER COMANDOS (CommandsReceiver) ---
            try:
                # O receive_and_convert agora deve retornar o objeto Protobuf 'Commands' cru
                # ou você adapta o CommandsReceiver para retornar a estrutura desserializada
                raw_data = self.rx.receiver_socket.recv(1024) 
                
                if raw_data:
                    # Desserializa
                    cmd_packet = command_pb2.Commands()
                    cmd_packet.ParseFromString(raw_data)
                    
                    # Coloca o PACOTE INTEIRO na fila da comunicação
                    if not self.udp_recv_queue.full():
                        self.udp_recv_queue.put(cmd_packet)
                        
            except BlockingIOError:
                pass # Nada chegou
            except Exception as e:
                print(f"[UDP Error] {e}")

            # Pequeno sleep para não fritar a CPU nessa thread de I/O
            time.sleep(0.001)


    def communicationThread(self):
        """
        Thread de comunicação do Emulador.
        Envia comandos, solicita status periódico e processa respostas.
        """
        # Import necessário para mapear o ID numérico (0,1,2) para o Address do PFOX
        from modules.communication.protocol.protocolHeader import Address

        STATUS_INTERVAL = getattr(self, "comm_send_interval", 3)  # em segundos
        STATUS_INTERVAL_MS = STATUS_INTERVAL * 1000  # converter para ms

        last_status_request = self.Timer.getElapsedTime()

        print("[Emulator] 🟢 Communication Thread iniciada")

        while self.cameraIsRunning:

            loop_start = self.Timer.getElapsedTime()

            # ---------------------------------------------------------
            # 0) Checar comunicação
            # ---------------------------------------------------------
            if not self.comm or not self.comm.is_connected() or self.comm._paused:
                time.sleep(0.2)
                continue
        
            # ---------------------------------------------------------
            # 1) Enviar comandos pendentes
            # ---------------------------------------------------------
            if self.comm and self.comm.is_connected() and self.comm.is_sending_enabled():
                try:
                    # Verifica a fila (se você usou 'udp_recv_queue' antes, ajuste o nome aqui)
                    while not self.udp_recv_queue.empty():
                        
                        # Pega o objeto 'Commands' (Protobuf) que veio da UDP Thread
                        proto_cmds = self.udp_recv_queue.get_nowait()
                        
                        # Verifica se é o objeto esperado (tem o campo robot_commands)
                        if hasattr(proto_cmds, 'robot_commands'):
                            for cmd in proto_cmds.robot_commands:
                                # cmd possui: id, yellowteam, wheel_left, wheel_right
                                
                                # Filtro de Time (Opcional: descomente se quiser filtrar por cor)
                                # if cmd.yellowteam != self.my_team_color_bool: continue

                                # Mapeamento: Protobuf ID (0, 1, 2) -> PFOX Address (ROBOT1, ROBOT2...)
                                target_id = Address(cmd.id + 1)

                                # Usa sua classe Communication para montar e enviar o pacote PFOX
                                self.comm.send_speed_command(
                                    robot_id=target_id,
                                    left_real=0,   # Sem feedback real por enquanto
                                    left_des=int(cmd.wheel_left),
                                    right_real=0,  # Sem feedback real por enquanto
                                    right_des=int(cmd.wheel_right)
                                )
                        
                except Exception as e:
                    print(f"[Emulator] ❌ Erro ao enviar comando: {e}")
                    # traceback.print_exc() # Descomente para debug profundo

            # ---------------------------------------------------------
            # 2) Envio periódico de heartbeat
            # ---------------------------------------------------------
            try:
                now = self.Timer.getElapsedTime()
                if now - last_status_request >= 3000:
                    self.comm.send_heartbeat()
                    last_status_request = now

            except Exception as e:
                print(f"[Emulator] ❌ Erro ao enviar heartbeat: {e}")
            
            # ---------------------------------------------------------
            # 3) Processar RX
            # ---------------------------------------------------------
            try:
                responses = self.comm.get_responses()
                for resp in responses:
                    if self.DEBUGA:
                        print(f"[Emulator RX] {resp}")

            except Exception as e:
                print(f"[Emulator] ❌ Erro ao processar RX: {e}")

            # ---------------------------------------------------------
            # 4) Fechamento do loop
            # ---------------------------------------------------------
            elapsed = self.Timer.getElapsedTime() - loop_start
            sleep_time = max(0, STATUS_INTERVAL - elapsed / 1000)
            
            # Sleep curto para garantir que não trave a thread, mas não consuma 100% CPU
            # Se STATUS_INTERVAL for muito longo, usamos um sleep menor fixo para manter responsividade
            if sleep_time > 0.01: 
                time.sleep(0.01) 
            else:
                time.sleep(sleep_time)
            
            loop_end = self.Timer.getElapsedTime()
            self.deque_send.append(loop_end - loop_start)
            

        print("[Emulator] 🔴 Communication Thread finalizada")
    # ==============================================================
    #  3. Filas, buffers e coleções
    # ==============================================================
    def _init_collections(self):
        """Configura filas e coleções auxiliares usadas no sistema."""
        self.commands_queue = queue.Queue(maxsize=1)
        self.received_data_queue = queue.Queue(maxsize=10)

        self.maxDeque = 4
        self.capture_deque = deque(maxlen=self.maxDeque)

        self.avg_window = 40 #quantidade de samples para média 

        # Criando deques para salvar os tempos
        self.deque_vision = deque(maxlen=self.avg_window)
        self.deque_proc = deque(maxlen=self.avg_window)
        self.deque_send = deque(maxlen=self.avg_window)
        self.deque_fps  = deque(maxlen=self.avg_window)

        # Entidades controladas
        self.field = None
        self.ball = None
        self.allies = [None, None, None]
        self.enemies = [None, None, None]

    def avg(self, dq):
        if len(dq) == 0:
            return 0
        return sum(dq)/len(dq)
    
    # ==============================================================
    #  4. Inicialização dos viewers
    # ==============================================================
    def _init_viewers(self):
        """Configura os elementos visuais do aplicativo (viewers e painéis)."""
        for viewer in [
            self.viewer,
            self.debugFieldViewer,
            self.debugObjectsViewer,
            self.debugPlayersViewer,
            self.debugTeamViewer,
            self.resultViewer,
            self.virtualResult,
        ]:
            if hasattr(viewer, "config"):
                viewer.config()

    # ==============================================================
    #  5. Controle e configuração lógica
    # ==============================================================
    def _init_control_system(self):
        """Inicializa o sistema de controle e define conteúdo dos robôs."""
        self.setContentRobots()


    # ==============================================================
    #  6. Timer de alta precisão
    # ==============================================================
    def _init_timers(self, App):
        """Cria o temporizador de alta precisão."""
        self.Timer = HighPrecisionTimer(self)
        self.frameTime = 0.0
        self.sendTime = 0.0
        self.FPStime = 0.0
        self.realTime = 0.0
        self.totalTime = 0.0

    def fill_deques_time(self):
        self.deque_vision.append(self.totalTime)
        self.deque_proc.append(self.frameTime)
        self.deque_send.append(self.sendTime)

    def erase_deques_times(self):
        self.deque_vision.clear()
        self.deque_proc.clear()
        self.deque_send.clear()


    # ==============================================================
    #  7. Informações do sistema operacional
    # ==============================================================
    def _init_system_info(self, App):
        """Armazena informações do sistema operacional."""
        self.OP = App.system
        self.OPrelease = App.release
        self.OPversion = App.version

    # ==============================================================
    #  8. GPU e CUDA
    # ==============================================================
    def _init_gpu_info(self):
        """Inicializa as variáveis relacionadas à GPU e CUDA."""
        self.hasCuda = False
        self.CudaDevice = None
        self.CudaDeviceVersion = None
        self.CUDAselected = False

    # ==============================================================
    #  9. Configurações da câmera
    # ==============================================================
    def _init_camera_settings(self):
        """Define parâmetros padrão da câmera."""
        self.FocusValue = 0
        self.FocusMode: FocusMode = FocusMode.AUTO

    # ==============================================================
    #  10. Comunicação e captura
    # ==============================================================
    def _init_communication(self):
        """Inicializa o objeto de comunicação."""
                # -------------------------
        # 🔹 Comunicação (via classe Communication)
        # -------------------------
        com_mode = self.comMode.lower()

        if com_mode == 'mqtt':
            broker = self.settingsTree.tree.item('I021', 'value')[0] if 'I021' in self.settingsTree.tree.get_children('') else 'localhost'
            port_str = self.settingsTree.tree.item('I022', 'value')[0] if 'I022' in self.settingsTree.tree.get_children('') else '1883'
            port = int(port_str) if port_str.isdigit() else 1883

            self.comm = Communication(use_mqtt=True, broker_address=broker, port=port)
            print(f"[EMULADOR] Comunicação configurada via MQTT → {broker}:{port}")

        elif com_mode == 'serial':
            port = self.serialPort or '/dev/ttyUSB0'
            self.comm = Communication(use_mqtt=False, serial_port=port)
            print(f"[EMULADOR] Comunicação configurada via Serial → {port}")

        else:
            # Nenhum modo de comunicação selecionado
            print('[EMULADOR] Comunicação desativada (nenhuma selecionada).')
            self.comm = None

        self.should_open_comm_window = (
            self.Mode == MODE_USB_CAM and 
            com_mode in ['mqtt', 'serial'] and
            self.comm is not None
        )
        
        if self.should_open_comm_window:
            print(f"[EMULADOR] Janela de comunicação será aberta - Modo: {com_mode}")
            
        # Inicializo a thread de comunicação
        self.comm_thread = threading.Thread(target=self.communicationThread, daemon=True)
        self.comm_thread.start()

        # Inicializo a thread de comunicação com a API de controle
        self.udp_thread = threading.Thread(target=self.strategyUDPThread, daemon=True)
        self.udp_thread.start()

        # 🔹 ABRE JANELA DE COMUNICAÇÃO (se necessário)
        if self.should_open_comm_window:
            # Agenda a abertura para depois da UI estar estável
            self.viewer.window.after(500, self._open_communication_window)

    def _init_capture_and_vision(self):
        """Cria a instância de captura e o sistema de visão."""
        self.capture = Capture(CaptureMode.DEFAULT, False)
        self.EConfig = EConfig()
        self.vs = VisionSystem(capture=self.capture, UseCuda=False, GPUType=None)

    #===================================================================
    def load_vars(self):
        """Carrega variáveis de configuração do emulador e ajusta o comportamento do sistema."""
        tree = self.settingsTree.tree
        get = lambda key: tree.item(key, 'value')[0]

        # -------------------------
        # 🔹 Entradas gerais
        # -------------------------
        self.CamUSB = int(get('I003'))
        self.ImgPath = get('I004')
        self.VideoPath = get('I005')
        self.UseMode = self.format_var(get('I006'))

        # -------------------------
        # 🔹 Parâmetros de processamento de imagem
        # -------------------------
        self.OffSetBord = int(get('I008'))
        self.OffSetErode = int(get('I009'))
        self.BINThresh = int(get('I00A'))
        self.MatrixTop = int(get('I00B'))
        self.FocusMode = self.format_var(get('I00C'))

        focus_str = get('I00D')
        self.FocusValue = float(focus_str) if focus_str else 0.0

        # -------------------------
        # 🔹 Campo
        # -------------------------
        self.fieldWidth = int(get('I00F'))
        self.fieldHeight = int(get('I010'))

        # -------------------------
        # 🔹 Cores
        # -------------------------
        color_keys = {
            "mainColor": 'I012',
            "player1Color1": 'I013',
            "player1Color2": 'I014',
            "player2Color1": 'I015',
            "player2Color2": 'I016',
            "player3Color1": 'I017',
            "player3Color2": 'I018',
            "enemiesMainColor": 'I019',
            "ballColor": 'I01A',
        }

        for attr, key in color_keys.items():
            setattr(self, attr, self.format_var(get(key)))

        # -------------------------
        # 🔹 Modo e execução
        # -------------------------
        self.debug_view = self.format_var(get('I01C'))
        self.comMode = self.format_var(get('I01D'))  # "mqtt" ou "serial" ou "nenhuma"
        self.serialPort = get('I01E')
        self.CUDAService = self.format_var(get('I01F'))
        self.EXECMode = self.format_var(get('I020'))

        # -------------------------
        # 🔹 Modo de debug
        # -------------------------
        debug_flag = str(self.debug_view).lower()
        self.DEBUGA = debug_flag == 'true'
        self.debug_view = self.DEBUGA

        # -------------------------
        # 🔹 Modo de uso (câmera, imagem, vídeo)
        # -------------------------
        mode_map = {
            'camera': MODE_USB_CAM,
            'imagem': MODE_IMAGE,
            'video': MODE_VIDEO_CAM,
        }
        self.Mode = mode_map.get(self.UseMode, MODE_DEFAULT)
        if self.Mode == MODE_DEFAULT:
            print('[EMULADOR] Valor inválido para modo de uso.')
            self.stop()

        # -------------------------
        # 🔹 CUDA
        # -------------------------
        self.CUDAselected = "Don't have cuda"

        # 🔹 Comunicação (via classe Communication)
        # -------------------------
        com_mode = self.comMode.lower()


        # -------------------------
        # 🔹 Foco da câmera
        # -------------------------
        focus_map = {
            'automatico': FocusMode.AUTO,
            'manual': FocusMode.MANUAL,
        }
        self._focusMode = focus_map.get(self.FocusMode, FocusMode.AUTO)

        # -------------------------
        # 🔹 Comunicação protobuff 
        # -------------------------
        self.ip_send = get('I022')
        self.port_send = int(get('I023'))
        self.ip_receive = get('I024')
        self.port_receive = int(get('I025'))

        # -------------------------
        # 🔹 Atualização do card
        # -------------------------
        self.infoCards.updateFuncs()

   
    def format_var(self, var):
        var = unidecode.unidecode(var)
        var = var.lower()
        return var

    def show_variables(self):
        msg=f"""
        [EMULADOR]
        ====Emulador===
        CamUSB          : {self.CamUSB}
        ImgPath         : {self.ImgPath}
        VideoPath       : {self.VideoPath}
        UseMode         : {self.UseMode}
        
        OffSetBord      : {self.OffSetBord}
        OffSetErode     : {self.OffSetErode}
        BINThresh       : {self.BINThresh}
        MatrixTop       : {self.MatrixTop}
        
        fieldWidth      : {self.fieldWidth}
        fieldHeight     : {self.fieldHeight}

        ally Color      : {self.mainColor}
        Goleiro CorP    : {self.player1Color1}
        Goleiro CorS    : {self.player1Color2}
        Atk1    CorP    : {self.player2Color1}
        Atk1    CorS    : {self.player2Color2}
        Atk2    CorP    : {self.player3Color1}
        Atk2    CorS    : {self.player3Color2}
        Enemies Color   : {self.enemiesMainColor}
        ballColor       : {self.ballColor}
        
        debug_view      : {self.debug_view}
        EXECMode        : {self.EXECMode}

        ComMode         : {self.comMode}
        SerialPort      : {self.serialPort}

        FocusMode       : {self.FocusMode}
        FocusValue      : {self.FocusValue}

        CUDAService     : {self.CUDAService}
        Mode            : {self.Mode}

        =================
        """.encode('utf-8')

        print(msg.decode('utf-8', errors='replace'))

    # Método para inicializar o processo utilizar o sistema de visão os resultados
    def init(self):
        print("[EMULADOR] Resetando estado anterior e configurando variáveis...")

        # 1. Reset completo do Vision System
        self.vs.ResetVs()
        if hasattr(self.vs, 'bmk'):
            self.vs.bmk.reset()

        # 2. Reset das entidades retidas no Emulador
        self.field = None
        self.ball = None
        self.allies = [None, None, None]
        self.enemies = [None, None, None]
        self.tproc_string = ""
        self.erase_deques_times()

        # 3. Esvaziar todas as filas concorrentes para não processar frames antigos
        for q in [self.ui_queue, self.comm_queue, self.udp_send_queue, self.udp_recv_queue, self.output_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # -----------------------------
        # Inicialização base
        # -----------------------------
        self.Timer.run()
        self.viewer.config()
        self.debugFieldViewer.config()
        self.infoCards.updateFuncs()

        def str_to_int_array(s: str) -> np.ndarray:
            return np.array(list(map(int, s.strip("[]").split())))

        # Dimensões e cores
        self.fieldDimensions = np.array([self.fieldWidth, self.fieldHeight])
        self.teamMainColor = str_to_int_array(self.mainColor)
        self.enemiesMainColor = str_to_int_array(self.enemiesMainColor)

        self.player1Colors = np.array([str_to_int_array(self.player1Color1), str_to_int_array(self.player1Color2)])
        self.player2Colors = np.array([str_to_int_array(self.player2Color1), str_to_int_array(self.player2Color2)])
        self.player3Colors = np.array([str_to_int_array(self.player3Color1), str_to_int_array(self.player3Color2)])
        self.playersAllColors = np.array([self.player1Colors, self.player2Colors, self.player3Colors])

        self.ballColor = str_to_int_array(self.ballColor)

        # -----------------------------
        # Criar e aplicar configuração do emulador
        # -----------------------------
        self.EConfig = EConfig(
            offSetWindow=self.OffSetBord,
            offSetErode=self.OffSetErode,
            dimMatrix=self.MatrixTop,
            Trashhold=self.BINThresh,
            FieldWidth=self.fieldWidth,
            FieldHeight=self.fieldHeight,
            ballColor=self.ballColor,
            allyColor=self.teamMainColor,
            enemyColor=self.enemiesMainColor,
            goalAllyColor1=self.player1Colors[0],
            goalAllyColor2=self.player1Colors[1],
            atk1AllyColor1=self.player2Colors[0],
            atk1AllyColor2=self.player2Colors[1],
            atk2AllyColor1=self.player3Colors[0],
            atk2AllyColor2=self.player3Colors[1],
            emulatorMode=self.Mode,
            timer=self.Timer,
            ip_receive=self.ip_receive,
            port_receive=self.port_receive,
            ip_send=self.ip_send,
            port_send=self.port_send
        )
        self.vs.SetConfigEmulator(self.EConfig)

        # -----------------------------
        # Inicialização conforme o modo
        # -----------------------------
        if self.Mode == MODE_USB_CAM:
            self._init_usb_mode()

        elif self.Mode == MODE_IMAGE:
            self._init_image_mode()

        elif self.Mode == MODE_VIDEO_CAM:
            self._init_video_mode()

        else:
            print('[EMULADOR] Entrada inválida.')
            self._reset_ui_and_stop()

    #==== Métodos auxiliares da inicialização ====
    def _init_usb_mode(self):
        print('[EMULADOR] Modo: Câmera USB')

        # Reset da captura
        self.capture.reset()
        self.capture.setMode(CaptureMode.CAM)

        # Tenta abrir a câmera
        if not self.capture.setIdCam(self.CamUSB):
            print("[EMULADOR] Falha ao abrir a câmera USB.")
            self._handle_camera_error()
            return

        # Configura foco da câmera
        self._configure_focus()

        # Marca câmera como ativa
        self.cameraIsRunning = True
        print("[CAPTURA] Iniciando thread da câmera...")

        # Inicia thread de captura e processamento
        print("[EMULADOR] Iniciando modo paralelo de captura e processamento...")

        self.cameraIsRunning = True
        self.Timer.run()

        # Inicia thread de visão (processamento pesado)
        self.vision_thread = threading.Thread(target=self.visionThread, daemon=True)
        self.vision_thread.start()

        self._init_communication()
        
        # Inicia loop da UI (Tkinter)
        self.viewer.window.after(0, self.updateUI)
    


    def _configure_focus(self):
        """Define o modo de foco de acordo com a configuração."""
        if not self.settingsTree._hasControlFocus:
            print("[EMULADOR] Árvore não detectou controle de foco. Aplicando padrão.")
            self._focusMode = FocusMode.AUTO

        if self._focusMode == FocusMode.AUTO:
            if not self.capture.setModeFocus(FocusMode.AUTO):
                print("[EMULADOR] Câmera não suporta foco automático.")
        else:
            if not self.capture.setModeFocus(FocusMode.MANUAL):
                print("[EMULADOR] Câmera não suporta foco manual.")
            else:
                self.capture.setFocusManual(self.FocusValue)
                print(f"[EMULADOR] Foco manual ajustado para {self.FocusValue}")

    def _open_communication_window(self):
        """Abre a janela de comunicação debug de forma assíncrona."""
        try:
            # ✅ CORREÇÃO: usar self.comm em vez de self.communication
            if (self.comm_debug_window is None and 
                self.comm is not None and  # ← CORRIGIDO
                self.cameraIsRunning):
                
                # Cria a janela de comunicação
                self.comm_debug_window = CommunicationDebugWindow(
                    self.app.root,  # Master é a janela principal
                    comm=self.comm,  # ← CORRIGIDO
                    on_close_ref_clear=self._on_comm_window_close
                )
                
                print("[EMULADOR] Janela de comunicação aberta com sucesso")
                
        except Exception as e:
            print(f"[EMULADOR] Erro ao abrir janela de comunicação: {e}")
            import traceback
            traceback.print_exc()

    def _on_comm_window_close(self, window_ref):
        """Callback quando a janela de comunicação é fechada."""
        self.comm_debug_window = None
        print("[EMULADOR] Janela de comunicação fechada")


    def _init_image_mode(self):
        print('[EMULADOR] Modo: Imagem estática')
        self.cameraIsRunning = False
        self.capture.reset()
        self.capture.setMode(CaptureMode.IMG)
        self.processImageNew()

    def _init_video_mode(self):
        print('[EMULADOR] Modo: Vídeo')
        self.btn_stop.pack(fill=BOTH, expand=1)

        self.capture.reset()
        self.capture.setMode(CaptureMode.VIDEO)
        self.capture.setVideoPath(self.VideoPath)

        try:
            # Puxa o FPS da câmera/vídeo
            fps = self.capture.CAM.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 30  # fallback

            self.delay = int(1000 / fps)
            
            # Puxa o total de frames e a duração do vídeo
            self.total_frames = int(self.capture.CAM.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = (self.total_frames / fps) if fps > 0 else 0.0

            print(f"[VIDEO] FPS: {fps:.2f} | Delay: {self.delay} ms")
            print(f"[VIDEO] Total de frames: {self.total_frames} | Duração estimada: {self.video_duration:.2f} s")

        except Exception as e:
            print(f"[VIDEO] Não foi possível obter dados do vídeo, usando padrões. Erro: {e}")
            self.delay = 33  # ~30 FPS
            self.total_frames = 0
            self.video_duration = 0.0

        self.cameraIsRunning = False
        self.video_active = True
        self.video_paused = False

        # Inicia a thread de processamento de vídeo
        self.video_thread = threading.Thread(target=self.videoProcessingThread, daemon=True)
        self.video_thread.start()

        # Inicia o loop de atualização da UI
        self.viewer.window.after(0, self.updateUI)

    # ==============================================================
    # Thread de processamento de vídeo (separada da UI)
    # ==============================================================
    def videoProcessingThread(self):
        print("[VIDEO THREAD] Iniciada.")
        
        try:
            while self.video_active:
                if self.video_paused:
                    time.sleep(0.05)
                    continue

                loop_start = time.time()

                try:
                    frame = self.capture.getImage()
                    
                    # Se o frame for None, o vídeo encerrou
                    if frame is None:
                        print("[VIDEO THREAD] Fim do vídeo alcançado.")
                        break

                    # --- Processamento ---
                    t_proc_start = time.time()
                    result = self.vs.ProcessImg(frame, self.DEBUGA)
                    
                    # Garantir que result seja uma imagem válida
                    if result is None:
                        print("[VIDEO THREAD] Aviso: result é None; usando o frame original para a UI.")
                        result = frame.copy()
                    elif isinstance(result, np.ndarray) and result.ndim == 2:
                        # Converter escala de cinza para BGR
                        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                    elif not isinstance(result, np.ndarray):
                        result = frame.copy()

                    # --- Atualizar string de tempos de processamento ---
                    campo = self.vs.bmk.get_avg("Campo")
                    bola = self.vs.bmk.get_avg("Bola")
                    players = self.vs.bmk.get_avg("Players")
                    self.tproc_string = f"{campo:.2f} / {bola:.2f} / {players:.2f}"
                    
                    objects = self.vs.GetObjects()
                    virtual = self.vs.virtualImg
                    t_proc_end = time.time()

                    self.frameTime = (t_proc_end - t_proc_start) * 1000.0
                    self.totalTime = (time.time() - loop_start) * 1000.0
                    self.realTime = self.Timer.getElapsedTime() / 1000.0

                    self.fill_deques_time()

                    # Atualiza objetos detectados
                    self.field = objects.get(ID_Objects.FIELD, self.field)
                    self.ball = objects.get(ID_Objects.BALL, self.ball)
                    self.allies = objects.get(ID_Objects.ALLIES, self.allies)
                    self.enemies = objects.get(ID_Objects.ENEMIES, self.enemies)

                    # Prepara dados para a UI
                    data = {
                        'frame': frame,
                        'result': result,
                        'virtual': virtual,
                        'field': self.field,
                        'ball': self.ball,
                        'allies': self.allies,
                        'enemies': self.enemies,
                        'vision_time': self.totalTime,
                        'proc_time': self.frameTime,
                        'real_time': self.realTime,
                        'fps': self.FPStime,
                        'timestamp': self.realTime
                    }

                    # Coloca na fila da UI (substitui se cheia)
                    if self.ui_queue.full():
                        try:
                            self.ui_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.ui_queue.put_nowait(data)

                except Exception as e:
                    print("[VIDEO THREAD] Erro:", e)
                    traceback.print_exc()
                    time.sleep(0.01)
                    continue

                # Controle de FPS
                elapsed = (time.time() - loop_start) * 1000
                sleep_time = self.delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time / 1000.0)

        finally:
            # Garantia de finalização limpa da thread
            self.video_active = False
            
            self.viewer.window.after(0, self.stop)

            print("[VIDEO THREAD] Finalizada.")

    def _handle_camera_error(self):
        """Trata erro de inicialização da câmera."""
        messagebox.showerror(
            "Erro de captura", 
            "O equipamento não tem permissão ou o índice de câmera é inválido."
        )
        if self.capture:
            self.capture.reset()

        if self.comm:
            self.comm.reset()

        self.cameraIsRunning = False
        self._reset_ui_and_stop()

    def _reset_ui_and_stop(self):
        """Reseta a interface e interrompe a execução."""
        self.Mode = MODE_DEFAULT
        self.viewer.default_mode()
        self.debugFieldViewer.default_mode()
        self.infoCards.update()
        self.Timer.stop()
        self.Timer.reset()
        self.stop()

    def stop(self):
        """
        Interrompe toda a execução do emulador...
        """
        print("[EMULATOR] Encerrando execução...")
        self.video_active = False
        self.video_paused = False

        # --- Reset do Sistema de Visão e variáveis internas ---
        if hasattr(self, "vs") and self.vs is not None:
            self.vs.ResetVs()
            if hasattr(self.vs, 'bmk'):
                self.vs.bmk.reset()

        self.field = None
        self.ball = None
        self.allies = [None, None, None]
        self.enemies = [None, None, None]

        # --- Fecha janela de comunicação ---
        if self.comm_debug_window is not None:
            try:
                self.comm_debug_window.destroy()
            except Exception as e:
                print(f"[EMULATOR] Erro ao fechar janela de comunicação: {e}")
            finally:
                self.comm_debug_window = None

        # --- sinaliza parada global ---
        self.cameraIsRunning = False

        # --- encerra threads (visão, vídeo, UDP, comunicação) ---
        if hasattr(self, "vision_thread") and self.vision_thread and self.vision_thread.is_alive():
            print("[EMULATOR] Aguardando thread de visão encerrar...")
            self.vision_thread.join(timeout=1.0)

        if hasattr(self, "video_thread") and self.video_thread and self.video_thread.is_alive():
            print("[EMULATOR] Aguardando thread de vídeo encerrar...")
            self.video_thread.join(timeout=1.0)
            self.video_thread = None

        if hasattr(self, "udp_thread") and self.udp_thread and self.udp_thread.is_alive():
            print("[EMULATOR] Aguardando thread de comunicação (UDP) encerrar...")
            self.udp_thread.join(timeout=1.0)

        if hasattr(self, "comm_thread") and self.comm_thread and self.comm_thread.is_alive():
            print("[EMULATOR] Aguardando thread de comunicação encerrar...")
            self.comm_thread.join(timeout=1.0)

        # --- para captura ---
        try:
            if hasattr(self, "capture") and self.capture is not None:
                self.capture.reset()
        except Exception as e:
            print("[EMULATOR] Erro ao resetar captura:", e)

        # --- para o temporizador ---
        try:
            self.Timer.stop()
            self.Timer.reset()
        except Exception as e:
            print("[EMULATOR] Erro ao parar Timer:", e)

        # --- limpa filas ---
        for q in [self.ui_queue, self.comm_queue, self.udp_send_queue, self.udp_recv_queue, self.output_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break

        # --- zera tempos ---
        self.totalTime = 0.0
        self.frameTime = 0.0
        self.sendTime = 0.0
        self.realTime = 0.0
        self.FPStime = 0
        self.errorCode = 0
        self.erase_deques_times()

        print("[EMULATOR] Execução finalizada com sucesso.")
        self.app.menu.att_node_id('I020', 'Parado')
        self.app.menu.save_to_json('config')
        self.app.update_control_buttons("Parado")
        
    # Métodos auxiliares do stop
    def _stop_capture_thread(self):
        """Para a thread de captura de forma segura."""
        if hasattr(self, "captureThread") and self.captureThread:
            try:
                self.captureThread.stop()
                print("[EMULATOR] Thread de captura parada com sucesso.")
            except Exception as e:
                print(f"[EMULATOR] Thread de captura forçada a terminar: {e}")
            finally:
                self.captureThread = None

    def _reset_capture(self):
        """Libera a câmera e limpa a fila de imagens."""
        if hasattr(self, "capture") and self.capture:
            try:
                self.capture.reset()
                print("[EMULATOR] Câmera liberada.")
            except Exception as e:
                print(f"[EMULATOR] Falha ao liberar câmera: {e}")

        # Limpa deque de imagens para evitar frames antigos
        self.capture_deque.clear()

    
    def _close_communication(self):
        """Fecha qualquer tipo de comunicação ativa (MQTT/Serial) via classe Communication."""
        # ✅ CORREÇÃO: usar self.comm em vez de self.communication
        if hasattr(self, "comm") and self.comm:
            try:
                print("[EMULATOR] Encerrando comunicação...")
                self.comm.reset()
                print("[EMULATOR] Comunicação encerrada com sucesso.")
            except Exception as e:
                print(f"[EMULATOR] Erro ao encerrar comunicação: {e}")
        else:
            print("[EMULATOR] Nenhuma comunicação ativa para encerrar.")
            
    def _reset_ui_by_mode(self):
        """Atualiza botões e estado visual conforme o modo atual."""
        print(f"[EMULADOR] Resetando UI para o modo: {self.Mode}")

        self.cameraIsRunning = False



    # ============================ | Métodos de processamento (imagem estática) | ==================
    # ========== // Trabalhar com um loop apenas // ==============

    #================================================================================
    #processando uma imagem utilizando o novo sistema de visão
    def processImageNew(self):
        print("[EMULADOR] Processando imagem: ",self.ImgPath)
        St1i = self.Timer.getElapsedTime()
        self.capture.setImagePath(self.ImgPath)
        self.frame = self.capture.getImage()
        self.debugFrame = self.frame.copy()

        #Método de RUN para imagem
        result = self.vs.ProcessImg(self.debugFrame, debug=self.DEBUGA)

        # Calcular médias de processamento para exibir
        campo_avg = self.vs.bmk.get_avg("Campo")
        bola_avg = self.vs.bmk.get_avg("Bola")
        players_avg = self.vs.bmk.get_avg("Players")
        self.tproc_string = f"{campo_avg:.2f} / {bola_avg:.2f} / {players_avg:.2f}"

        #retornando valores
        St2i = self.Timer.getElapsedTime()

        #Exibindo dados em tela
        self.viewer.show(self.frame)

        #imagens de debug
        if(self.DEBUGA == True):
            binary_treat, binaryBall, binaryPlayers, binaryTeam = self.vs.GetDebugImages()
            self.debugFieldViewer.show(binary_treat)
            self.debugObjectsViewer.show(binaryBall)
            self.debugPlayersViewer.show(binaryPlayers)
            self.debugTeamViewer.show(binaryTeam)
        else:
            self.debugFieldViewer.clear()
            self.debugObjectsViewer.clear()
            self.debugPlayersViewer.clear()
            self.debugTeamViewer.clear()
            
        #imagens de resultado
        self.resultViewer.show(result)
        self.virtualResult.show(self.vs.virtualImg)
        
        # puxando informações do sistema de visão
        self.allies     = self.vs.allyTeam
        self.enemies    = self.vs.enemyTeam

        #Adicionando conteúdos
        self.setContentRobots()

        #resetando configurações já que é modo imagem
        self.vs.ResetVs()

        self.totalTime = (St2i - St1i)                       #tempo em mili 
        self.frameTime = self.totalTime
        #segundos
        
        self.realTime = self.Timer.getElapsedTime() / 1000

        self.FPStime = int(1000/self.totalTime)

        #atualizo informações na interface
        self.fill_deques_time()
        self.infoCards.update()
        self.vs.bmk.reset()
        self.erase_deques_times()

        self.app.menu.att_node_id('I020', 'Parado')
        self.app.menu.save_to_json('config')
        self.app.update_control_buttons("Parado")
    
    
    def _safe_show(self, viewer, image):
        if image is not None and isinstance(image, np.ndarray) and image.size > 0:
            viewer.show(image)
        else:
            viewer.clear()

    # ==============================================================
    # Controle de pausa e retomada (para vídeo)
    # ==============================================================
    def pause(self):
        """Pausa o loop de vídeo (se aplicável)."""
        if self.Mode == MODE_VIDEO_CAM:
            self.video_paused = True
            print("[VIDEO] Pausado.")
        else:
            # Para outros modos, implementar se necessário (ex.: pausar threads)
            pass

    def resume(self):
        """Retoma o loop de vídeo após pausa."""
        if self.Mode == MODE_VIDEO_CAM and self.video_paused:
            self.video_paused = False
            print("[VIDEO] Retomado.")
            # A thread continuará naturalmente pois a flag foi desativada

    #Adicionar conteúdo dos robÔs
    def setContentRobots(self):
        for i in range(3):
            if self.allies[i] is not None:
                self.cards[i].set_content(self.allies[i].id, self.allies[i].detected, self.allies[i].position, self.allies[i].radius, self.allies[i].image)
            else:
                self.cards[i].set_content("#0", False, ["0.0", "0.0"], "0.0", None)
                
        for i in range(3):
            if self.enemies[i] is not None:
                self.cards[i+3].set_content(self.enemies[i].id, self.enemies[i].detected, self.enemies[i].position, self.enemies[i].radius, self.enemies[i].image)
            else:
                self.cards[i+3].set_content("#0", False, ["0.0", "0.0"], "0.0", None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()