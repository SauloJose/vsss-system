from ui.viewer import MyViewer
from modules.VisionSys.components.objects import *
from imports import *
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Treeview, Scrollbar, Entry, Style
from tkinter import simpledialog, messagebox, filedialog
import cv2
import serial
import serial.tools.list_ports
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import json
import base64
import unidecode

class settingsMenu(Frame):
    def __init__(self, App, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.app = App
        self.tree = Treeview(self, columns=('Valor',))
        self.tree.heading('#0',text="Variável")
        self.tree.heading('Valor', text='Valor')
        self.tree.column('Valor',stretch=False,minwidth=50, width=100)
        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.pack(fill=BOTH, expand=True)
        self.nodes={}
        self._hasControlFocus = None

        #variável para evitar que mais de uma instância seja acionada na threeview
        # Assim ela só aciona uma instância por vês
        self._hasChild = False 
        ''' Variável interna que informa que a classe já tem uma instância em uso, para não permitir mais de uma instância de janelas filho.'''
        # self.color_editables = ['Cor principal', 'J1 Cor 1', 'J1 Cor 2', 'J2 Cor 1', 'J2 Cor 2', 'J3 Cor 1', 'J3 Cor 2', 'Cor inimigos', 'Cor da bola']


    #Adicionando um nó
    def add_node(self, parent,id, name, value):
        if not self.tree:
            return
        node_id = self.tree.insert(parent,"end", text=name,values=(value,))
        self.nodes[node_id] = value
        return node_id

    def att_node_id(self,id, value):
        self.tree.set(id,'Valor',value)
        self.nodes[id] = value
        self.tree.item(id,tags=())

    def on_double_click  (self, event):
        editable_items = ['Câmera USB','Vídeo Path','Cor','Cor Principal T1', 'Cor Principal T2','T1_robo 1','T1_robo 2','T1_robo 3','T2_robo 1','T2_robo 2','T2_robo 3','Borda da janela','offSet da Erosão','Binarização Threshold','Dim. Matriz TOPHAT', 'Largura (eixo x)', 'Comprimento (eixo y)','IP de Envio','Porta de Envio','IP de Recebimento','Porta de Recebimento']
        #color_editables = ["Calibração das Cores"]
        color_editables = ['Cor principal', 'Goal Cor 1', 'Goal Cor 2', 'Atk1 Cor 1', 'Atk1 Cor 2', 'Atk2 Cor 1', 'Atk2 Cor 2', 'Cor inimigos', 'Cor da bola']
        item = self.tree.focus()
        if item:
            if self.tree.item(item,'text') in color_editables:
                #Caso isso aconteça, ele irá exibir um frame para modificar as cores e verificar o código
                if not self._hasChild:
                    self.open_color_pick_window(item)

            elif self.tree.item(item, 'text') == 'Modo de Uso':
                self.open_mode_window(self.tree, item)

            elif self.tree.item(item, 'text') == 'Imagem Path':
                self.load_file(self.tree, item)

            elif self.tree.item(item, 'text') == 'Vídeo Path':
                self.load_file(self.tree, item)

            elif self.tree.item(item, 'text') == 'Debug':
                self.open_select_window(self.tree, item)
            
            elif self.tree.item(item, 'text') == 'CUDA':
                self.open_select_window(self.tree, item)
                
            # selecionando modo de conexão do computador
            elif self.tree.item(item, 'text') == 'Comunicação':
                if not self._hasChild:
                    self.open_window_connection(self.tree, item)

            #seleciona o foco da câmera
            elif self.tree.item(item, 'text') == 'Modo do Foco':
                if not self._hasChild:
                    self.open_focus_pick_window(self.tree, item)

            elif self.tree.item(item, 'text') in editable_items:
                self.tree.item(item, tags=('edit',))
                name = str(self.tree.item(item)['text'])
                title = "Editar Variável "+name
                txt = "Adicione o novo valor da variável "+name
                entry = simpledialog.askstring(title, txt)
                if entry is not None:
                    self.tree.set(item,'Valor',entry)
                    self.nodes[item] = entry
                self.tree.item(item,tags=())
            else:
                #Em caso negativo, ele não apenas irá retornar à configuração padrão
                pass


    def load_file(self, tree, item):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.tree.set(item,'Valor', file_path)
            self.nodes[item] = file_path

    def open_mode_window(self, tree, item):
        self.item = item
        self.tree = tree

        new_window = Toplevel(self.tree)
        new_window.title("Seleção de modo")
        self.root = new_window

        try:
            if(self.app.system == 'Windows'):
                self.root.iconbitmap('src/data/icon.ico')
            elif(self.app.system =='Linux'):
                self.root.iconbitmap('src/data/icon.ico')
            else:
                self.root.iconbitmap('src/data/icon.ico')
        except:
            print("[APP]: Problemas em acessar o ícone")

        def update_value():
            mode_picked = pick_var.get()
            self.tree.set(item,'Valor',mode_picked)
            self.nodes[item] = mode_picked

            new_window.destroy()




        options = ['camera', 'imagem', 'video']
        texts = ['Câmera', 'Imagem', 'Vídeo']

        pick_var = StringVar()

        for i in range(3):
            radiobutton = ttk.Radiobutton(new_window, text=texts[i], variable=pick_var, value=options[i])
            radiobutton.pack(padx=10, pady=5)

        select_button = Button(new_window, text="Confirmar", command=update_value)
        select_button.pack(pady=10)

    #Janela se seleção true false
    def open_select_window(self, tree, item):
 
        self.item = item
        self.tree = tree

        name = self.tree.item(item)['text']

        new_window = Toplevel(self.tree)
        new_window.title("Seleção de modo")
        self.root = new_window

        try:
            if(self.app.system == 'Windows'):
                self.root.iconbitmap('src/data/icon.ico')
            elif(self.app.system =='Linux'):
                self.root.iconbitmap('src/data/icon.ico')
            else:
                self.root.iconbitmap('src/data/icon.ico')
        except:
            print("[APP]: Problemas em acessar o ícone")

        def update_value():
            mode_picked = pick_var.get()
            self.tree.set(item,'Valor',mode_picked)
            self.nodes[item] = mode_picked

            #informa que foi liberado a janela
            new_window.destroy()

        options = ['true', 'false']
        texts = ['Ativado', 'Desativado']

        pick_var = StringVar()

        
        #label
        txt = "Modo da Variável "+str(name)
        label = Label(new_window, bg = "white", text=txt)
        label.pack()

        for i in range(2):
            radiobutton = ttk.Radiobutton(new_window, text=texts[i], variable=pick_var, value=options[i])
            radiobutton.pack(padx=10, pady=5)

        select_button = Button(new_window, text="Confirmar", command=update_value)
        select_button.pack(pady=10)
    
    
    #Selecionar tipo de conexão
    def open_window_connection(self,tree, item):
        #define filho
        self._hasChild = True 

        #definindo a janela
        new_window = Toplevel(self.tree)
        new_window.title ("Seleção tipo de comunicação")
        self.root = new_window

        #adicionando ícone
        try:
            if(self.app.system == 'Windows'):
                self.root.iconbitmap('src/data/icon.ico')
            elif(self.app.system =='Linux'):
                self.root.iconbitmap('src/data/icon.ico')
            else:
                self.root.iconbitmap('src/data/icon.ico')
        except:
            print("[APP]: Problemas em acessar o ícone")
                # Calcula as dimensões da janela
        
        #frame principal
        frame = ttk.Frame(self.root)
        frame.pack(padx = 10, pady = 10)

        #label de seleção de comunicação
        label_comm = ttk.Label(frame, text="Selecione o tipo de comunicação:")
        label_comm.grid(row=0, column=0, padx=5, pady=5)

        # Combobox para seleção de comunicação
        combobox_comm = ttk.Combobox(frame, values=["MQTT", "SERIAL","Nenhuma"])
        combobox_comm.grid(row=0, column=1, padx=5, pady=5)

        # Label de seleção de porta serial
        label_serial = ttk.Label(frame, text="Selecione a porta serial:")
        label_serial.grid(row=1, column=0, padx=5, pady=5)
        label_serial.grid_remove()  # Inicialmente oculto

        # Combobox para seleção de porta serial
        combobox_serial = ttk.Combobox(frame)
        combobox_serial.grid(row=1, column=1, padx=5, pady=5)
        combobox_serial.grid_remove()  # Inicialmente oculto)

        #exibe as seriais conectadas
        def populate_serial_ports():
            serial_ports = [port.device for port in serial.tools.list_ports.comports()]
            combobox_serial["values"] = serial_ports
            combobox_serial["state"] = "readonly"  # Define o Combobox como somente leitura


        #mostra as opções de entrada serial
        def show_serial_options(event):
            selected_comm = combobox_comm.get()
            if selected_comm == "SERIAL":
                label_serial.grid()
                combobox_serial.grid()
                populate_serial_ports()

            else:
                label_serial.grid_remove()
                combobox_serial.grid_remove()

        #atualiza as portas seriais em tempo real
        def update_serial_ports():
            populate_serial_ports()
            # Chama esta função novamente após 1000 milissegundos (1 segundo)
            self.root.after(1000, update_serial_ports)
        
        #chama esse método para atualizar
        update_serial_ports()

        #destroi janela
        def destroy_window():            
            #libera funcionalidades
            self._hasChild = False
            self.root.destroy()

        #adiciona um protocolo para fechar de maneira segura a janela
        new_window.protocol("WM_DELETE_WINDOW",destroy_window)

        #confirma seleção
        def on_confirm():
            selected_comm = combobox_comm.get()
            selected_port = combobox_serial.get() if selected_comm == "SERIAL" else None

            if selected_comm == "MQTT":
                self.tree.set(item, 'Valor', selected_comm)
                self.tree.set('I01E', 'Valor', "")
                self.nodes[item] = selected_comm
                self.nodes['I01E'] = ""
                destroy_window()
            elif selected_comm == "SERIAL":
                if selected_comm == "SERIAL" and not selected_port:
                    messagebox.showerror("Erro", "Selecione uma porta serial!")
                else:
                    self.tree.set(item, 'Valor', selected_comm)
                    self.tree.set('I01E', 'Valor', selected_port)
                    self.nodes[item] = selected_comm
                    self.nodes['I01E'] = selected_port
                    destroy_window()

            elif selected_comm == "Nenhuma":
                    self.tree.set(item, 'Valor', selected_comm)
                    self.tree.set('I01E', 'Valor', "")
                    self.nodes[item] = selected_comm
                    self.nodes['I01E'] = ""
                    destroy_window()
            else:
                    #Provavelmente foi um erro, então mantem o padrão
                    self.tree.set(item, 'Valor', "Nenhuma")
                    self.tree.set('I01E', 'Valor', "")
                    self.nodes[item] = "Nenhuma"
                    self.nodes['I01E'] = ""
                    destroy_window()



        #definindo callbacks
        combobox_comm.bind("<<ComboboxSelected>>", show_serial_options)

        # Botão de voltar
        button_back = ttk.Button(frame, text="Voltar", command=destroy_window)
        button_back.grid(row=2, column=0, padx=5, pady=5)

        # Botão de confirmar
        button_confirm = ttk.Button(frame, text="Confirmar", command=on_confirm)
        button_confirm.grid(row=2, column=1, padx=5, pady=5)
        
    #selecionar calibração das cores
    def open_focus_pick_window(self, tree, item):
        #aciona flag
        self._hasChild = True 

        self.focusMode = self.tree.item('I00C','value')[0]
        self.focusValue = self.tree.item('I00D','value')[0]
        
        self.mode = self.tree.item('I006','value')[0]
        self.camPath = int(self.tree.item('I003','value')[0])
        

        #estado configurado no menu
        self._modeFocusWindows = 'AUTOMATICO'
        self._focusValue = 0 

        if self.mode == "camera":
            #Apenas executa esse modo de cor da janela caso o emulador não já esteja sendo utilizado
            # verifico se a câmera funciona                
            #verifico se suporta controle automático de foco
            # Suporta controle automático de foco
            self._hasControlFocus = True 
            #gera janela
            new_window = Toplevel(self.tree, bg="white")
            new_window.title("Calibração do Foco")
            self.root = new_window
            #adicionando ícones
            try:
                if(self.app.system == 'Windows'):
                    self.root.iconbitmap('src/data/icon.ico')
                elif(self.app.system =='Linux'):
                    self.root.iconbitmap('src/data/icon.ico')
                else:
                    self.root.iconbitmap('src/data/icon.ico')
            except:
                print("[APP]: Problemas em acessar o ícone")

            #definindo funções que vou utilizar
            def close_window():
                self._focusValue = 0
                self._hasControlFocus = False
                self._modeFocusWindows = None
                self.root.destroy()     # destruo
                self._hasChild = False
                
            #função que confirma e salva os valores
            def Confirm_stats():
                if(self._modeFocusWindows == 'AUTOMATICO'):
                    self.tree.set(item, 'Valor', self._modeFocusWindows)
                    self.tree.set('I00D', 'Valor', "")
                    self.nodes[item] = self._modeFocusWindows
                    self.nodes['I00D'] = ''
                elif (self._modeFocusWindows == 'MANUAL'):
                    #puxando valor configurado
                    self._focusValue = focusScale.get()
                    self._focusValue = focusScale.get()
                    self.tree.set(item, 'Valor', 'MANUAL')
                    self.tree.set('I00D', 'Valor', self._focusValue)
                    self.nodes[item] = 'MANUAL'
                    self.nodes['I00D'] = self._focusValue
                else: #supor que é automático
                    self.tree.set(item, 'Valor', 'AUTOMATICO')
                    self.tree.set('I00D', 'Valor', "")
                    self.nodes[item] = 'AUTOMATICO'
                    self.nodes['I00D'] = ''
                
                close_window()
            
            #definindo evento
            def on_select(event):
                self._modeFocusWindows = modecombobox.get()
                if(self._modeFocusWindows == 'AUTOMATICO'):
                    print("Está no foco automático")
                    focusScale['state'] = DISABLED
                else:  
                    print("Está no foco manual")
                    focusScale['state'] = NORMAL
                    self._focusValue = focusScale.get()
            
            #Adiciona um protocolo a new_window para desligar a câmera
            self.root.protocol("WM_DELETE_WINDOW",close_window)
            #Criando os labels
            #Escolhe o modo de foco da imagem
            modoFrame = Frame(self.root, bg="white")
            modoFrame.pack()
            modoTextLabel = Label(modoFrame, text="Configuração do foco da câmera: ", bg="white")
            modoTextLabel.grid(row=0, column=0, padx=5,pady=5)
            modecombobox = ttk.Combobox(modoFrame, values =['AUTOMATICO', 'MANUAL'],state="readonly")
            modecombobox.set("AUTOMATICO")
            modecombobox.grid(row=0, column=1, padx=5, pady=5)
            
            #verifica se a câmera tem controle de 
            #adicionando evento no combobox
            modecombobox.bind("<<ComboboxSelected>>", on_select)
            #escolhe o valor do foco da imagem
            focusFrame = Frame(self.root, bg="white")
            focusFrame.pack()
            focusText = Label(focusFrame, text="Focus: ", bg="white")
            focusText.grid(row=0, column=0, padx=5,pady=5)
            focusScale = Scale(focusFrame, from_=0, to=255, resolution=0.1, orient=HORIZONTAL, bg="white")
            focusScale.grid(row=0, column=1, padx=5, pady=5)
            focusScale['state'] = DISABLED
            
            #label explicativo
            i2= Frame(self.root,  bg="white")
            i2.pack()
            i2text = Label(i2, text="Deseja Confirmar?", bg="white")
            i2text.pack()
            #label para realizar a ação de confirmar ou voltar a tela
            actionFrame = Frame(self.root, bg="white")
            actionFrame.pack()
            btnVoltar = Button(actionFrame, text="Voltar", command = close_window)
            btnVoltar.grid(row=0, column=0, padx=5, pady=5)
            btnConfirm = Button(actionFrame, text = "Confirmar", command=Confirm_stats )
            btnConfirm.grid(row=0, column=1, padx=5, pady=5)
            #Inicia captura de imagens
            #CaptureShow()
        else:
            messagebox.showwarning("Cuidado!", "Tem que estar selecionado o modo de captura por câmera")


    def open_color_pick_window(self, item):
        self._hasChild = True
        self.mode = self.tree.item('I006', 'value')[0]
        self.imgPath = self.tree.item('I004', 'value')[0]
        self.camPath = self.tree.item('I003', 'value')[0]
        self.videoPath = self.tree.item('I005', 'value')[0]

        win = Toplevel(self.tree)
        win.title("Seleção de Cores")
        self.root = win
        win.resizable(False, False)

        # ícone
        try:
            win.iconbitmap('src/data/icon.ico')
        except:
            print("[APP]: Problemas em acessar o ícone")

        # captura de mídia (se necessário)
        self.cap = None
        self.first_frame = None

        if self.mode == "camera":
            try:
                self.cap = cv2.VideoCapture(int(self.camPath))
            except:
                self.cap = cv2.VideoCapture(0)

        elif self.mode == "video":
            cap_temp = cv2.VideoCapture(self.videoPath)
            ret, frame_temp = cap_temp.read()
            if ret:
                self.first_frame = frame_temp
            cap_temp.release()

        def close_window():
            if self.cap: self.cap.release()
            win.destroy()
            self.cap = None
            self._hasChild = False
        win.protocol("WM_DELETE_WINDOW", close_window)

        # --- ESTRUTURA DE LAYOUT (ESQUERDA / DIREITA) ---
        main_container = Frame(win)
        main_container.pack(padx=15, pady=15)

        left_frame = Frame(main_container)
        left_frame.pack(side=LEFT, padx=(0, 15), fill=Y)

        right_frame = Frame(main_container)
        right_frame.pack(side=RIGHT)

        # --- DIMENSÕES DAS IMAGENS ---
        img_w, img_h = 600, 350  # Imagem ampliada

        # --- FUNÇÕES DE SUPORTE E CONTROLE ---
        def rgb_to_hex(rgb):
            return f'#{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}'

        def get_color_from_sliders():
            return np.array([hue_scale.get(), saturation_scale.get(), value_scale.get()], dtype=np.uint8)

        def update_color(val=None):
            hsv = get_color_from_sliders()
            bgr = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0]
            color_display.configure(bg=rgb_to_hex(bgr))

        def pick_color():
            hsv = get_color_from_sliders()
            self.tree.set(item, 'Valor', hsv)
            self.nodes[item] = hsv
            close_window()

        def string_to_int_array(array):
            values = array.strip("[]").split()
            return list(map(int, values))

        # --- CONTEÚDO DO PAINEL ESQUERDO (CONTROLES) ---
        Label(left_frame, text="Cor Selecionada", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        color_display = Label(left_frame, text="", width=15, height=2, bg="gray", relief="ridge")
        color_display.pack(pady=(0, 15))

        current_hsv = string_to_int_array(self.tree.item(item, 'value')[0])

        hue_scale = Scale(left_frame, from_=0, to=179, orient="horizontal", label="Matiz (H)", length=250, command=update_color)
        hue_scale.set(current_hsv[0]); hue_scale.pack(pady=5)

        saturation_scale = Scale(left_frame, from_=0, to=255, orient="horizontal", label="Saturação (S)", length=250, command=update_color)
        saturation_scale.set(current_hsv[1]); saturation_scale.pack(pady=5)

        value_scale = Scale(left_frame, from_=0, to=255, orient="horizontal", label="Valor (V)", length=250, command=update_color)
        value_scale.set(current_hsv[2]); value_scale.pack(pady=5)

        # Botões de Ação na Esquerda
        btn_frame = Frame(left_frame)
        btn_frame.pack(pady=(20, 0))
        Button(btn_frame, text="Selecionar Cor", command=pick_color, bg="#4CAF50", fg="white", font=("Arial", 9, "bold")).pack(fill=X, pady=2)
        Button(btn_frame, text="Voltar", command=close_window).pack(fill=X, pady=2)

        # --- CONTEÚDO DO PAINEL DIREITO (IMAGENS) ---
        Label(right_frame, text="Clique na imagem para capturar a cor", font=("Arial", 9, "italic")).pack()

        original_label = Label(right_frame, text="Imagem Original", cursor="crosshair")
        original_label.pack(pady=(5, 10))

        masked_label = Label(right_frame, text="Imagem Filtrada")
        masked_label.pack(pady=5)

        # Tolerâncias do filtro
        hue_tol = 6 if self.tree.item(item, 'text') == 'Cor da bola' else 10
        sat_tol, val_tol = 50, 50

        # --- CLIQUE NA IMAGEM (CONTA-GOTAS) ---
        def on_click(event):
            if hasattr(capture_frame, 'last_frame') and capture_frame.last_frame is not None:
                # Limita as coordenadas ao tamanho da imagem renderizada
                x = min(max(0, event.x), img_w - 1)
                y = min(max(0, event.y), img_h - 1)

                # Pega o pixel BGR (NumPy usa formato [linha/y, coluna/x])
                bgr_pixel = np.uint8([[capture_frame.last_frame[y, x]]])

                # Converte para HSV
                hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]

                # Atualiza os sliders
                hue_scale.set(hsv_pixel[0])
                saturation_scale.set(hsv_pixel[1])
                value_scale.set(hsv_pixel[2])

                update_color()

        original_label.bind("<Button-1>", on_click)

        # --- LOOP DE CAPTURA E EXIBIÇÃO ---
        def capture_frame():
            frame = None

            if self.mode == "imagem":
                frame = cv2.imread(self.imgPath)
            elif self.mode == "video":
                if hasattr(self, 'first_frame') and self.first_frame is not None:
                    frame = self.first_frame.copy()
            elif self.mode == "camera" and self.cap:
                ret, frame = self.cap.read()
                if not ret: return
            else:
                return

            if frame is None:
                return

            # Redimensiona para o novo tamanho ampliado
            frame = cv2.resize(frame, (img_w, img_h))
            capture_frame.last_frame = frame.copy()

            # --- CORREÇÃO DO ERRO DE OVERFLOW / DTYPE ---
            # Converte valores H, S, V para 'int' do Python antes de somar/subtrair
            h, s, v = map(int, get_color_from_sliders())

            # Garante tipo np.uint8 e aplica limites válidos de HSV no OpenCV
            lower = np.array([
                max(0, h - hue_tol),
                max(0, s - sat_tol),
                max(0, v - val_tol)
            ], dtype=np.uint8)

            upper = np.array([
                min(179, h + hue_tol),  # No OpenCV o Hue vai até 179
                min(255, s + sat_tol),
                min(255, v + val_tol)
            ], dtype=np.uint8)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            masked = cv2.bitwise_and(frame, frame, mask=mask)

            # Atualiza os quadros Tkinter
            for img, lbl in [(frame, original_label), (masked, masked_label)]:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                lbl.img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
                lbl.config(image=lbl.img)

            win.after(10, capture_frame)

        capture_frame()

    #Adquirindo os dados com o get_tree_data, é uma função recursiva
    def get_tree_data(self, node_id=''):
        if node_id == '':
            children = self.tree.get_children()
        else:
            children = self.tree.get_children(node_id)

        data = {}
        for child in children:
            item_id = child
            name = self.tree.item(child, 'text')
            value = self.tree.item(child, 'values')
            data[item_id] = {"name": name, "value": value, "children": self.get_tree_data(child)}
        
        return data
    
    def save_to_json(self, filename):
        data = self.get_tree_data()
        with open(filename + '.json', "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

    # Método para carregar a TreeView a partir de um arquivo JSON
    def load_from_json(self, node):
        name = node.get('name')
        value = node.get('value')
        children = node.get('children')

        item_id = self.tree.insert('', 'end', text=name, values=value)

        # self.tree.tag_configure('custom_color', background='lightblue')
        # self.tree.item(item_id, tags=('custom_color'))

        if children:
            for child in children.values():
                self.load_from_json_recursive(item_id, child)
    
    def rgb_to_hex(rgb):
        return f'#{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}'
    
    #Carrega dados de um arquivo json
    def load_from_json_recursive(self, parent_id, node):
        name = node.get('name')
        value = node.get('value')
        children = node.get('children')

        item_id = self.tree.insert(parent_id, 'end', text=name, values=value)


        # if self.tree.item(item_id,'text') in self.color_editables:

        #     bgr_color = cv2.cvtColor(np.uint8([[value]]), cv2.COLOR_HSV2BGR)
        #     rgb_color = self.rgb_to_hex(bgr_color[0, 0, :])

        #     self.tree.tag_configure('custom_color', background=rgb_color)
        #     self.tree.item(item_id, tags=('custom_color'))
    
        # self.tree.tag_configure('custom_color', background='lightblue')
        # self.tree.item(item_id, tags=('custom color'))
        
        if children:
            for child in children.values():
                self.load_from_json_recursive(item_id, child)

    #Carrega configurações para um arquivo json
    def load_from_json_file(self, filename):
        with open(filename, 'r',encoding='utf-8') as file:
            data = json.load(file)

        root_node = data.get('I001')
        self.load_from_json(root_node)
