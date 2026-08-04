import tkinter as tk
from imports import *
from PIL import Image, ImageTk, ImageGrab

class Card:
    def __init__(self, master, title, color):
        self.master = master
        self.title = title
        self.color = color

        self.frame = Frame(master, bg="white")
        self.frame.grid(row=0, column=0)

        self.contentFrame = Frame(self.frame, bg='white')
        self.contentFrame.grid(row=0, column=0)
        self.imageFrame = Frame(self.frame, bg='white')
        self.imageFrame.grid(row=0, column=1)

        self.titleLabel = Label(self.contentFrame, text=title, font=("Helvetica", 11), bg="white")
        self.titleLabel.pack()
        self.idLabel = Label(self.contentFrame, text="", font=("Helvetica", 8), bg="white")
        self.idLabel.pack()
        self.positionLabel = Label(self.contentFrame, text="", font=("Helvetica", 8), bg="white")
        self.positionLabel.pack()

        self.canvas = Canvas(self.imageFrame, width=50, height=50)
        self.canvas.pack()

        self.image_path = "src/data/images/dark_screen.png"
        self.load_image()

    def load_image(self):
        try:
            image = tk.PhotoImage(file=self.image_path)
            self.canvas.create_image(40, 40, anchor=tk.CENTER, image=image)
            self.canvas.image = image
        except tk.TclError:
            print("Erro ao carregar a imagem.")

    def show_image(self, image):
        cv2Img = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        pilImage = Image.fromarray(cv2Img)
        
        # Obter as dimensões atuais da imagem
        width, height = pilImage.size
        
        # Definir o tamanho máximo desejado
        max_width = 50
        max_height = 50
        
        # Calcular a porcentagem de redução com base nas dimensões atuais
        percent_reduction = min(max_width / width, max_height / height)
        
        # Calcular o novo tamanho da imagem
        new_width = int(width * percent_reduction)
        new_height = int(height * percent_reduction)
        
        # Redimensionar a imagem
        #resized = pilImage.resize((new_width, new_height), Image.ANTIALIAS)
        resized = pilImage.resize((new_width, new_height), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(image=resized)
        
        # Calcular as coordenadas para centralizar a imagem
        x = (max_width - new_width) // 2
        y = (max_height - new_height) // 2
    
        self.canvas.create_image(x,y, anchor=NW, image=self.image)

    def set_content(self, id, status, position, radius, image):
        try:
            # Garante que posição e raio sejam válidos
            if position is not None and len(position) >= 2:
                x = f"{float(position[0]):.1f}"
                y = f"{float(position[1]):.1f}"
            else:
                x, y = "0.00", "0.00"

            r = f"{float(radius):.1f}" if radius is not None else "0.00"

            # Define texto e cor conforme o status
            if status:
                status_text = "Detectado"
                color = "lightgreen"
            else:
                status_text = "Não Detectado"
                color = "lightcoral"

            # Atualiza labels
            self.idLabel.config(text=f"{status_text}", bg=color)
            self.positionLabel.config(text=f"POS: [{x} | {y}] cm\nr = {r} cm")

            # Exibe imagem, se houver
            if image is not None:
                self.show_image(image)

        except Exception as e:
            print(f"[Card][set_content] Erro ao atualizar card ({id}): {e}")



class CardInfos:
    def __init__(self, master, title):
        self.master = master
        self.title = title
        self.emulator = None
        
        # Configurando a expansão do frame
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)


        #Informações que ele irá exibir
        #pegará as informações de tempo de frame, tempo de envio
        #e código de erro
        self.FPS = None
        self.frameTime = None
        self.sendTime = None
        self.timer = None
        self.errorEmulate = None
        self.Mode = None
        self.totalTime = None
        
        #Criando tabs para configurar entre o modo desempenho ou modo aplicações
        self.tabs = ttk.Notebook(self.master)
        self.tab1 = Frame(self.tabs)
        self.tab2 = Frame(self.tabs)

        self.tab1.configure(background="white")
        self.tab2.configure(background="white")

        self.tabs.add(self.tab1, text="Desempenho")
        self.tabs.add(self.tab2, text="Funcionalidades")

        
        self.tabs.grid(row=0, column=1, sticky="nsew")

        #configurando a aba Desempenho
        self.texts = {
            "FPS:": tk.StringVar(),                     # quadros por segundo
            "Vision. (ms):": tk.StringVar(),            # tempo para processar imagem
            "(F/B/P).(ms):": tk.StringVar(),            # tempos de campo / bola / players
            "Proc. (ms):": tk.StringVar(),               # Tempo total para processar e imprimir
            "Envio (ms):": tk.StringVar(),              # tempo para enviar
            "Timer (s):": tk.StringVar(),               # timer de alta precisão do emulador
            "Modo:": tk.StringVar()                     # modo de emulação atual
        }

        for i, (label_text, var) in enumerate(self.texts.items()):
            label = tk.Label(self.tab1, text=label_text, bg="white", font=("Arial", 8, "bold"))
            label.grid(row=i + 1, column=0, sticky="w")
            text = tk.Label(self.tab1, textvariable=var, bg="white", font=("Arial", 8, "normal"))
            text.grid(row=i + 1, column=1, sticky="w")
            

        #configurando a aba funcionalidades
        #configurando a aba Desempenho
        self.funcs = {
            "Sys:": tk.StringVar(),                 # Qual sistema está executando o código
            "CUDA:": tk.StringVar(),                # se tem suporte ao cuda
            "cuDev:": tk.StringVar(),               # qual o serviço cuda
            "COM:": tk.StringVar(),                 # Se foi enviada a mensagem
            "SC1:": tk.StringVar(),                 # instrução do carro 1
            "SC2:": tk.StringVar(),                 # instrução do carro 2
            "SC3:": tk.StringVar(),                 # instrução do carro 3
        }

        for i, (label_text, var) in enumerate(self.funcs.items()):
            label2 = tk.Label(self.tab2, text=label_text, bg="white", font=("Arial", 8, "bold"))
            label2.grid(row=i + 1, column=0, sticky="w")
            text2 = tk.Label(self.tab2, textvariable=var, bg="white", font=("Arial", 8, "normal"))
            text2.grid(row=i + 1, column=1, sticky="w")


    #Atualiza informação
    def updateInfo(self, variable, value):
        #verifica se é um valor válido
        if isinstance(value, (int, float)):
            if(variable == 'FPS:' or variable == 'Error Code:'):
                strValue = f"{value:.0f}"
            else:
                strValue = f"{value:.2f}"
        elif isinstance(value,(str)):
            strValue = value
        else:
            # Lidar com o caso em que value não é um número
            strValue = "N/A"

        if variable in self.texts:
            self.texts[variable].set(strValue)
        else:
            print("[CARD]: Variável não encontrada: [",variable,"]","[",value,"]")
        
    #Atualiza contínuamente as informações acessando o objeto emulador
    def update(self):
        # Média do tempo total (ms)
        avg_total = self.emulator.avg(self.emulator.deque_vision)

        # Cálculo correto do FPS real
        fps = 1000 / avg_total if avg_total > 0 else 0

        # Atualiza FPS usando o valor calculado
        self.updateInfo("FPS:", fps)

        # Atualiza restantes com suas médias
        self.updateInfo("Vision. (ms):", self.emulator.avg(self.emulator.deque_vision))
        self.updateInfo("(F/B/P).(ms):", self.emulator.tproc_string)
        self.updateInfo("Proc. (ms):", self.emulator.avg(self.emulator.deque_proc))
        self.updateInfo("Envio (ms):", self.emulator.avg(self.emulator.deque_send))

        # Valores diretos
        self.updateInfo("Timer (s):", self.emulator.realTime)
        self.updateInfo("Modo:", self.conversionMode(self.emulator.Mode))

        
    #atualizando funcionalidades utilizadas para a emulação
    def updateFunc(self, variable, value):
        #Envia apenas strings
        strValue = value

        #Trata a informação
        if(variable == 'CUDA:'):
            if(value == True):
                strValue = "Disponível"
                strValue += ' / '
                if(self.emulator.CUDAselected == True):
                    strValue += "ON"
                else:
                    strValue += "OFF"
            else:
                strValue = "Indisponível / OFF"
        elif (variable == 'COM:'):
            if(value == True):
                if(self.emulator.hasMqtt):
                    strValue = "MQTT"
                elif(self.emulator.hasSerial):
                    strValue = "Serial entry / "
                    strValue += str(self.emulator.serialPort)
                else:
                    #provávelmente um erro
                    strValue = "Sem conexão"
            else:
                strValue = "Sem conexão"
        else:
            strValue = str(value)


        #Verifica se está na biblioteca que preciso
        if variable in self.funcs:
            self.funcs[variable].set(strValue)
        else:
            print("Variável não encontrada.")
    
    #Atualiza tudo de uma vez
    def updateFuncs(self):
        self.updateFunc("Sys:", str(self.emulator.OP+' '+self.emulator.OPversion))
        self.updateFunc("CUDA:",self.emulator.hasCuda)
        self.updateFunc("cuDev:",self.emulator.CudaDevice)
        self.updateFunc("COM:",False)
        self.updateFunc("SC1:",self.emulator.commands)
        self.updateFunc("SC2:"," N/A ")
        self.updateFunc("SC3:"," N/A ")



    #função para setar um mestre
    def setMaster(self, emulator):
        self.emulator = emulator   
        self.update()
    
    #converte o inteiro no modo de execução
    def conversionMode(self, n):
        id = int(n)
        str = ""
        
        if(id == 1):
            str = "Camera"
        elif (id == 2):
            str = "Imagem"
        elif (id == 3):
            str = "Video"
        else:
            str = "Default"
            
        return str
        
if __name__ == "__main__":
    print("Módulo sendo executado como funcao principal")
