'''
    @GNOMIO: Aplicação principal do sistema VSSS. Responsável por criar e gerenciar a interface gráfica,
    inicializar os módulos e coordenar a interação entre eles.

    Versão: v3.0.1
    Última modificação: 14/02/2024
    Autor: Saulo (update)

    Patch Notes v3.0.1:
    - Adicionado suporte a processamento paralelo com threads separadas
    - Nova UI com melhor organização e feedback visual
    - Melhorias na estabilidade da captura de câmera
    - Suporte a múltiplos modos de visualização
    - Sistema de debug aprimorado
    - Correções de memory leaks
'''
#=============================================================

from imports import *
from ui.settingsMenu import settingsMenu
from ui.viewer import MyViewer, WindowsViewer
from modules.emulator.emulator import Emulator
from ui.cards import *
from modules.communication.communication import *
from modules.VisionSys.components.objects import *

execution = False

class App:
    def __init__(self):
        root = Tk()
        self.root = root    
        self.menu = None
        self.menuTop= None
        
        #estilo do tema do tkinter
        '''        
        theme_choi
        
        ce = "breeze"

        style =ThemedStyle(root)
        style.theme_use(theme_choice)'''

        #verifica qual o sistema operacional
        self.system = platform.system()
        self.release = platform.release()
        self.version = platform.version()

        #variáveis de tamanho da tela do computador
        self.screen_height = None
        self.screen_width = None

        #tamanho da janela
        self.width = 1100
        self.height = 750

        self.configure_window() #self.window()

        self.create_main_frames() #self.frame_window()
        self.create_settings_frame() #self.frame_config()

        self.IdFrame = 2
        self.widgets_settings_frame()
        self.widgets_emulate_frame()
        self.widgets_images_frame()

        self.viewer  = MyViewer(self.tab1)
        self.debugField = MyViewer(self.tab2)
        self.debugObject = MyViewer(self.tab3)
        self.debugPlayers = MyViewer(self.tab4)
        self.debugTeam = MyViewer(self.tab5)
        self.result = MyViewer(self.tab6)
        self.virtualVision = MyViewer(self.tab51)   #Mostra a virtualização da imagem

        self.cards = []
        self.names = ['Goleiro (A)', 'Atacante 1 (A)','Atacante 2 (A)','Goleiro (E)', 'Atacante 1 (E)','Atacante 2 (E)' ]
        for name in self.names:
            self.card = Card(self.players_infos, "{}".format(name), "white")
            self.cards.append(self.card)

        for i, card in enumerate(self.cards):
            card.frame.grid(row=i // 3, column=i % 3, padx=10, pady=10)

        #bloco de informações do emulador
        self.infosEmulator = CardInfos(self.infos_emulate,'Informações')
        
        #criando o objeto emulador
        self.emulator = Emulator(self)

        self.infosEmulator.setMaster(self.emulator)
        
        #já inicia carregando as informações antigas do emulador
        self.emulator.load_vars()

        #configurando um menu na interface
        self.configure_menu()

        #inicia looping principal
        root.mainloop()


    #configarando menu

    #configurando a janela do projeto
    def configure_window(self):
        #propriedades
        self.root.title("PINBOT - VSSS")
        self.root.configure(background="#dfe3ee")
        self.root.geometry("1100x750")
        self.root.resizable(False, False)

        #colocando ícone
        try:
            if(self.system == 'Windows'):
                self.root.iconbitmap('src/data/icon.ico')
            elif(self.system =='Linux'):
                self.root.iconbitmap('src/data/icon.ico')
            else:
                self.root.iconbitmap('src/data/icon.ico')
        except:
            print("[APP]: Problemas em acessar o ícone")
                # Calcula as dimensões da janela
            
        #posicionando a janela no centro
        self.center_window()
    
    def create_main_frames(self):
        self.settings_frame = Frame(self.root, bg="white")
        self.settings_frame.place(relx=0.01,rely=0.015, relwidth=0.29,relheight=0.97)
        
        self.images_frame = Frame(self.root,bg="white")
        self.images_frame.place(relx=0.31,rely=0.015, relwidth=0.68,relheight=0.68)
        
        self.vision_data_frame = Frame(self.root, bg="black")
        self.vision_data_frame.place(relx=0.31,rely=0.705, relwidth=0.68,relheight=0.28)
        
        #criando um frame do tipo grid
        self.players_infos = Frame(self.vision_data_frame, bg="white")
        self.players_infos.place(relx=0,rely=0, relwidth=0.76,relheight=1)
        
        #criando um frame para as informações
        self.infos_emulate = Frame(self.vision_data_frame, bg="white")
        self.infos_emulate.place(relx=0.76,rely=0, relwidth=1,relheight=1)

        
    def create_settings_frame(self):
        self.set_settings_frame=Frame(self.settings_frame, bg="white")
        self.set_settings_frame.place(relx=0,rely=0, relwidth=1,relheight=0.90)

        self.emulate_frame=Frame(self.settings_frame, bg="white")
        self.emulate_frame.place(relx=0,rely=0.90, relwidth=1,relheight=0.10)

    def widgets_settings_frame(self):
        self.menu = settingsMenu(self, self.set_settings_frame)
        self.menu.pack(fill=BOTH, expand=True)

        self.exec = execution

        if(self.exec):
            print('Configurações de Fábrica')
            self.build_tree_menu()
            self.exec = False
            try:
                self.menu.save_to_json('config')
            except:
                print('por algum motivo, não foi possível salvar o json')
        else:
            try:
                print('Carregando arquivo da memoria')
                self.menu.load_from_json_file('config.json')
                print("carregadas sucesso.")
            except:
                print("Não foi possível puxar o arquivo da memória. Será reiniciado o sistema")
                self.build_tree_menu()
                self.exec = True
                print('Configurações de Fábrica')
                self.menu.save_to_json('config')

        self.save_label = Label(self.set_settings_frame, text="Salvar as configurações")
        self.save_label.place(relx=0.1, rely=0.85, relwidth=0.8, relheight=0.05)
        self.save_label.config(fg="black",bg="white")
        self.save_button = Button(self.set_settings_frame, text='Salvar', command=self.save_config)
        self.save_button.place(relx=0.3, rely=0.90, relwidth=0.4, relheight=0.05)

    def widgets_emulate_frame(self):
        self.label_run_emulate = Label(self.emulate_frame, text="Estado da Emulação")
        self.label_run_emulate.pack(fill=X, pady=2)

        # Container para fixar os 3 botões sempre visíveis lado a lado
        self.btn_container = Frame(self.emulate_frame, bg="white")
        self.btn_container.pack(fill=BOTH, expand=True, padx=5, pady=2)

        self.btn_run = Button(self.btn_container, text="Executar", bg="darkgray", command=self.init_emulate)
        self.btn_run.pack(side=LEFT, fill=BOTH, expand=True, padx=2)

        self.btn_pause = Button(self.btn_container, text="Pausar", bg="darkgray", command=self.pause_emulate)
        self.btn_pause.pack(side=LEFT, fill=BOTH, expand=True, padx=2)

        self.btn_stop = Button(self.btn_container, text="Parar", bg="darkgray", command=self.stop_emulate)
        self.btn_stop.pack(side=LEFT, fill=BOTH, expand=True, padx=2)

        # Inicializa as permissões de clique com base no estado "Parado"
        self.update_control_buttons("Parado")

    def update_control_buttons(self, state):
        """Gerencia quais botões podem ser clicados com base no Modo de Uso e Estado"""
        
        # Busca dinâmica e segura para saber o modo de uso selecionado na interface
        use_mode = "camera"
        try:
            tree_data = self.menu.get_tree_data()
            def find_mode(data):
                if isinstance(data, dict):
                    if 'UseMode' in data: return data['UseMode']
                    for v in data.values():
                        res = find_mode(v)
                        if res: return res
                elif isinstance(data, list):
                    for item in data:
                        res = find_mode(item)
                        if res: return res
                return None
            found = find_mode(tree_data)
            if found: use_mode = str(found).lower()
        except:
            pass

        is_image_mode = "imagem" in use_mode or "image" in use_mode
        is_video_mode = "video" in use_mode or "vídeo" in use_mode

        # --- REGRA PARA MODO IMAGEM ---
        if is_image_mode:
            self.btn_run.config(text="Executar", state=NORMAL)
            self.btn_pause.config(state=DISABLED)
            self.btn_stop.config(state=DISABLED)
            return

        # --- REGRAS PARA MODO VÍDEO OU WEBCAM ---
        if state == "Parado":
            self.btn_run.config(text="Executar", state=NORMAL)
            self.btn_pause.config(state=DISABLED)
            self.btn_stop.config(state=DISABLED)
        
        elif state == "Em execução.":
            self.btn_run.config(text="Executar", state=DISABLED) # Bloqueia o executar se já está rodando
            self.btn_pause.config(state=NORMAL if is_video_mode else DISABLED) # Pausar exclusivo para vídeo
            self.btn_stop.config(state=NORMAL)
            
        elif state == "Pausado":
            self.btn_run.config(text="Retomar", state=NORMAL) # Libera o executar (como Retomar)
            self.btn_pause.config(state=DISABLED)
            self.btn_stop.config(state=NORMAL)

    def widgets_images_frame(self):
        self.tabs = ttk.Notebook(self.images_frame)
        self.tab1 = Frame(self.tabs)
        self.tab2 = Frame(self.tabs)
        self.tab3 = Frame(self.tabs)
        self.tab4 = Frame(self.tabs)
        self.tab5 = Frame(self.tabs)
        self.tab51 = Frame(self.tabs)
        self.tab6 = Frame(self.tabs)

        self.tab1.configure(background="black")
        self.tab2.configure(background="black")
        self.tab3.configure(background="black")
        self.tab4.configure(background="black")
        self.tab5.configure(background="black")
        self.tab51.configure(background='black')
        self.tab6.configure(background="black")

        self.tabs.add(self.tab1, text="Imagem")
        self.tabs.add(self.tab2, text="Debug campo")
        self.tabs.add(self.tab3, text="Debug bola")
        self.tabs.add(self.tab4, text="Debug jogadores")
        self.tabs.add(self.tab5, text="Debug time")
        self.tabs.add(self.tab51, text = "Virtual")
        self.tabs.add(self.tab6, text="Resultado")

        self.tabs.place(relx=0, rely=0, relwidth=1, relheight=1)

    def build_tree_menu(self):

        SysVision = self.menu.add_node('','SysVisPinbot','Sistema de Visão', value='')

        EntryData=self.menu.add_node(SysVision,'EntryData','Entrada de dados', value='')
        self.menu.add_node(EntryData,'CamPth','Câmera USB', value='---')
        self.menu.add_node(EntryData,'ImagePath','Imagem Path', value='---')
        self.menu.add_node(EntryData,'VideoPath','Vídeo Path', value='---')
        self.menu.add_node(EntryData,'UseMode','Modo de Uso', value='Camera')

        AlgRec = self.menu.add_node(SysVision,'Algoritm','Calibração do algorítmo',value='')
        self.menu.add_node(AlgRec,'offsetW','Borda da janela', value='10')
        self.menu.add_node(AlgRec,'offSetErode','offSet da Erosão', value='3')
        self.menu.add_node(AlgRec,'Threshold','Binarização Threshold', value='245')
        self.menu.add_node(AlgRec,'DimMatrx','Dim. Matriz TOPHAT', value='25')    
        self.menu.add_node(AlgRec,'FocoCam','Modo do Foco', value='Automatico')
        self.menu.add_node(AlgRec,'FocoValue','Valor do Foco', value='')


        FieldSize=self.menu.add_node(SysVision,'fieldDimensions','Dimensões do campo (cm)', value='')
        self.menu.add_node(FieldSize,'fieldWidth','Largura (eixo x)', value='150')
        self.menu.add_node(FieldSize,'fieldHeight','Comprimento (eixo y)', value='130')

        CalColor=self.menu.add_node(SysVision,'ColorCalibration','Calibração das Cores', value='')
        self.menu.add_node(CalColor,'mainColor','Cor principal', value='[ 90 128 128]')
        self.menu.add_node(CalColor,'j1Color1','Goal Cor 1', value='[ 90 128 128]')
        self.menu.add_node(CalColor,'j1Color2','Goal Cor 2', value='[ 90 128 128]')
        self.menu.add_node(CalColor,'j2Color1','Atk1 Cor 1', value='[ 90 128 128]')
        self.menu.add_node(CalColor,'j2Color2','Atk1 Cor 2', value='[ 90 128 128]')
        self.menu.add_node(CalColor,'j3Color1','Atk2 Cor 1', value='[ 90 128 128]')
        self.menu.add_node(CalColor,'j3Color2','Atk2 Cor 2', value='[ 90 128 128]')
        self.menu.add_node(CalColor,'enemyColor','Cor inimigos', value='[ 90 128 128]')   
        self.menu.add_node(CalColor,'ballColor','Cor da bola', value='[9 93 220]')
        
        ConfigEmulator=self.menu.add_node(SysVision,'EmulatorConfig','Configurações do Emulador', value='')
        self.menu.add_node(ConfigEmulator,'Debug','Debug', value='False')
        self.menu.add_node(ConfigEmulator,'Comunicação','Comunicação', value='nenhuma')
        self.menu.add_node(ConfigEmulator,'Porta Serial','Porta Serial',value = ' ')
        self.menu.add_node(ConfigEmulator,'CUDA','CUDA', value='False')
        self.menu.add_node(ConfigEmulator, 'ExectState', 'Estado de Execução', value='Parado')

        protobuffConfig = self.menu.add_node(SysVision,'ProtobuffConfig','Configurações Protobuff', value='')
        self.menu.add_node(protobuffConfig,'I022','IP de Envio', value='127.0.0.1')
        self.menu.add_node(protobuffConfig,'I023','Porta de Envio', value='54321')
        self.menu.add_node(protobuffConfig,'I024','IP de Recebimento', value='127.0.0.1')
        self.menu.add_node(protobuffConfig,'I025','Porta de Recebimento', value='54322')

    def save_config(self):
        self.data = self.menu.get_tree_data()
        with open('config.json', 'w',encoding="utf-8") as f:
            json.dump(self.data, f,ensure_ascii=False)
        print('[APP] config salvas com sucesso!')
        self.emulator.load_vars()

    def init_emulate(self):
        print("\n[APP] Emulação Iniciada/Retomada")
        self.emulator.load_vars()
        self.menu.save_to_json('config')

        # Se for modo vídeo e estiver pausado, apenas retoma
        if self.emulator.Mode == MODE_VIDEO_CAM and self.emulator.video_paused:
            self.emulator.resume()
        else:
            self.emulator.init()

        # ATUALIZAÇÃO: Só força "Em execução" se não for modo imagem
        if self.emulator.Mode != MODE_IMAGE:  # MODE_IMAGE deve estar importado (ex: de imports)
            self.menu.att_node_id('I020', 'Em execução.')
            self.menu.save_to_json('config')
            self.update_control_buttons("Em execução.")
        else:
            # Para modo imagem, o estado já foi definido como "Parado" pelo processImageNew
            # Garantimos que os botões estejam no estado correto (já foi chamado lá)
            # Mas se não foi, chamamos aqui também
            self.update_control_buttons("Parado")

    def pause_emulate(self):
        print("\n[APP] Emulação Pausada")
        if hasattr(self.emulator, 'pause'):
            self.emulator.pause()
        else:
            print("[AVISO] Método 'pause' não implementado na classe Emulator do backend.")
            
        self.menu.att_node_id('I020','Pausado')
        self.menu.save_to_json('config')
        
        self.update_control_buttons("Pausado")

    def stop_emulate(self): 
        self.emulator.stop()
        self.menu.att_node_id('I020','Parado')
        self.menu.save_to_json('config')
        
        self.update_control_buttons("Parado")

    #pegar as informações da tela
    def get_screen_resolution(self):
        #funcionando no windows
        if(self.system == 'Windows'):
            user32 = ctypes.windll.user32
            self.screen_width = user32.GetSystemMetrics(0)
            self.screen_height = user32.GetSystemMetrics(1)
        
        #funcionando no linux
        elif(self.system == 'Linux'):
            # Executar o comando xrandr e obter a saída
            output = subprocess.check_output(['xrandr']).decode('utf-8')

            # Expressão regular para encontrar as dimensões da tela
            pattern = r'\b(\d+)x(\d+)\+\d+\+\d+\b'

            # Procurar as dimensões da tela na saída do xrandr
            match = re.search(pattern, output)

            # Se for encontrada uma correspondência, extrair as dimensões
            if match:
              
                self.screen_width, self.screen_height = map(int, match.groups())
                print("Largura:", self.screen_width)
                print("Altura:", self.screen_height)
        else:
             print("Não foi possível encontrar as dimensões da tela.")


    #centralizando a janela na tela
    def center_window(self):
        try:
            if(self.system =='Windows'):
                hwnd = ctypes.windll.user32.FindWindowW(u"Shell_traywnd", None)
                rect = ctypes.wintypes.RECT()
                ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
                taskbar_height = rect.bottom - rect.top
            else:
                taskbar_height = 0 
        except:
            taskbar_height = 0

        self.get_screen_resolution()

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        x = (self.screen_width // 2) - (self.width // 2)
        y = (self.screen_height // 2) - (self.height // 2) - (taskbar_height//2)

        # Impede que a janela seja redimensionada
        self.root.resizable(False, False)

        # Defina a geometria da janela
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
    
    #configurando a mensagem de erro
    def show_error_message(self, message):
        messagebox.showerror("Erro", message)
    
    #configurando o menu
    def configure_menu(self):
        #criando o menu
        self.menuTop = Menu(self.root)
        self.root.config(menu=self.menuTop)

        #criando menu de informações
        self.infoMenu = Menu(self.menuTop, tearoff=0)
        self.menuTop.add_cascade(label='Informações',menu=self.infoMenu)
        self.infoMenu.add_command(label="Abrir informações", command=None)
        self.infoMenu.add_separator()

        #criando menus arquivo
        self.arqMenu = Menu(self.menuTop, tearoff=0)
        self.menuTop.add_cascade(label='Arquivos',menu=self.arqMenu)
        self.arqMenu.add_command(label="Abrir arquivo de backup", command=None)
        self.arqMenu.add_command(label="Salvar backup", command=None)
        self.arqMenu.add_command(label="Retornar às configurações de fábrica", command=None)
        self.arqMenu.add_separator()

        #criando menu de jogadores
        self.control= Menu(self.menuTop, tearoff=0)
        self.menuTop.add_cascade(label='Comunicação',menu=self.control)
        self.control.add_command(label="Abrir janela de comunicação",command = self.emulator._open_communication_window())
        self.control.add_separator()

        #criando menu de ajuda
        self.helpMenu = Menu(self.menuTop, tearoff=0)
        self.menuTop.add_cascade(label='Ajuda',menu=self.helpMenu)
        self.helpMenu.add_command(label='Sequência de uso', command=None)
        self.helpMenu.add_separator()


if __name__ == "__main__":
    app = App()