from imports import *

#Classe de player para o aplicativo
class MyViewer:
    def __init__(self, window: Frame):
        #print("[VIEWER] Viewer criado")
        self.window = window

    #Método para configurar o viewer para a função definida
    def config(self):
        #print("[VIEWER] Viewer em modo de configuração")
        #Configurand o o canvas
        self.canvas = Canvas(self.window, width=780, height=510)
        self.canvas.place(relx=0, rely=0)
        self.canvas.create_rectangle(0, 0, 748, 510, fill="black")

        
    #Resetando as configurações do viewer para forma padrão
    def resetConfig(self):
        #print("[VIEWER] Resetando configuração de exibição")
        #Retornado ao modo padrão
        self.default_mode()
    
    #Método de [VIEWER]viewer padrão sem exibir
    def default_mode(self):
        #print("[VIEWER] Viewer não configurado")
        self.canvas = Canvas(self.window, width=780, height=480)
        self.canvas.place(relx=0, rely=0)
        self.canvas.create_rectangle(0, 0, 748, 510, fill="black")
    
    def show(self, image):
        cv2Img = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        pilImage = Image.fromarray(cv2Img)
        
        # Obter as dimensões atuais da imagem
        width, height = pilImage.size
        
        # Definir o tamanho máximo desejado
        max_width = 748
        max_height = 480
        
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
    
    def clear(self):
        """Apaga tudo e coloca fundo preto."""
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 748, 510, fill="black")

class WindowsViewer:
    def __init__(self, window: Frame):
        #print("[VIEWER] Viewer criado")
        #Frame que irá ser utilizado
        self.window = window

    #Método para configurar o viewer para a função definida
    def config(self):
        wCanva = int(720/3)
        hCanva = int(480/2)
        #print("[VIEWER] Viewer em modo de configuração")
        #Configurand o o canvas
        self.canvas = Canvas(self.window, width=780, height=510)
        #self.canvas = Canvas(self.window, width=200, height=100)
        self.canvas.place(relx=0, rely=0)
        self.canvas.create_rectangle(0, 0, 748, 510, fill="red")
        #self.canvas.create_rectangle(0, 0, 500, 300, fill="red")

        self.canvas1 = Canvas(self.window, width=wCanva, height=hCanva)
        self.canvas1.place(relx=0, rely=0)
        self.canvas1.create_rectangle(0, 0, wCanva, hCanva, fill="gray")
        
        self.canvas2 = Canvas(self.window, width=wCanva, height=hCanva)
        self.canvas2.place(relx=(1/3), rely=0)
        self.canvas2.create_rectangle(0, 0, wCanva, hCanva, fill="gray")

        self.canvas3 = Canvas(self.window, width=wCanva, height=hCanva)
        self.canvas3.place(relx=(2/3), rely=0)
        self.canvas3.create_rectangle(0, 0, wCanva, hCanva, fill="gray")

        self.canvas4 = Canvas(self.window, width=wCanva, height=hCanva)
        self.canvas4.place(relx=0, rely=0.5)
        self.canvas4.create_rectangle(0, 0, wCanva, hCanva, fill="gray")

        self.canvas5 = Canvas(self.window, width=wCanva, height=hCanva)
        self.canvas5.place(relx=(1/3), rely=0.5)
        self.canvas5.create_rectangle(0, 0, wCanva, hCanva, fill="gray")

        self.canvas6 = Canvas(self.window, width=wCanva, height=hCanva)
        self.canvas6.place(relx=(2/3), rely=0.5)
        self.canvas6.create_rectangle(0, 0, wCanva, hCanva, fill="gray")

    #Resetando as configurações do viewer para forma padrão
    def resetConfig(self):
        #print("[VIEWER] Resetando configuração de exibição")
        #Retornado ao modo padrão
        self.default_mode()
    
    #Método de [VIEWER]viewer padrão sem exibir
    def default_mode(self):
        #print("[VIEWER] Viewer não configurado")
        self.canvas = Canvas(self.window, width=780, height=480)
        self.canvas.place(relx=0, rely=0)
        self.canvas.create_rectangle(0, 0, 748, 510, fill="black")
    
    def show(self, image):
        tamanho = len(image)

        pathImg = 'src/data/images/dark_screen.png'
        imgagemPadrao = cv2.imread(pathImg)
        cv2Img = cv2.cvtColor(imgagemPadrao, cv2.COLOR_BGR2RGBA)
        pilImage = Image.fromarray(cv2Img)
        self.img = ImageTk.PhotoImage(image=pilImage)
        self.listaDeImagens = [self.img, self.img, self.img, self.img, self.img, self.img]
        self.x = 100
        self.y = 100

        if(len(image) > 0):
            for i in range(len(image)):

                if(image[i] is not None):
                    cv2Img = cv2.cvtColor(image[i], cv2.COLOR_BGR2RGBA)
                    pilImage = Image.fromarray(cv2Img)
                    
                    # Obter as dimensões atuais da imagem
                    width, height = pilImage.size
                    
                    # Definir o tamanho máximo desejado
                    #max_width = 748
                    #max_height = 480

                    max_width = int(748/3)
                    max_height = int(480/2)

                    # Calcular a porcentagem de redução com base nas dimensões atuais
                    percent_reduction = min(max_width / width, max_height / height)
                    
                    # Calcular o novo tamanho da imagem
                    new_width = int(width * percent_reduction)
                    new_height = int(height * percent_reduction)
                    
                    # Redimensionar a imagem
                    #resized = pilImage.resize((new_width, new_height), Image.ANTIALIAS)
                    resized = pilImage.resize((new_width, new_height), Image.LANCZOS)
                    self.image = ImageTk.PhotoImage(image=resized)
                    self.listaDeImagens[i] = self.image

                    # Calcular as coordenadas para centralizar a imagem
                    self.x = (max_width - new_width) // 2
                    self.y = (max_height - new_height) // 2
            
        if(len(self.listaDeImagens) == 6):
            self.canvas1.create_image(self.x, self.y, anchor=NW, image=self.listaDeImagens[0])
            self.canvas2.create_image(self.x, self.y, anchor=NW, image=self.listaDeImagens[1])
            self.canvas3.create_image(self.x, self.y, anchor=NW, image=self.listaDeImagens[2])
            self.canvas4.create_image(self.x, self.y, anchor=NW, image=self.listaDeImagens[3])
            self.canvas5.create_image(self.x, self.y, anchor=NW, image=self.listaDeImagens[4])
            self.canvas6.create_image(self.x, self.y, anchor=NW, image=self.listaDeImagens[5])

    def clear(self):
        """Apaga tudo e coloca fundo preto."""
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 748, 510, fill="black")