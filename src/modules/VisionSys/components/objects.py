# ==========================================================================================
#  Variáveis, objetos e constantes padrões do código
#==========================================================================================
'''
    @GNOMIO: Descrição de objetos, cosntantes e padrão do código para ser utilizado.

    Versão: v3.0.1
    Última modificação: 14/02/2024
    Autor: Saulo (update)

    Patch Notes v3.0.1:
    - Apenas atualizada algumas classes.
'''
import numpy as np
import time
from ui.viewer import MyViewer
import threading
import queue
import tkinter 
from collections import deque, defaultdict
from ui.settingsMenu import *

from tkinter import *
from tkinter import ttk
from tkinter.ttk import Treeview, Scrollbar, Entry, Style
from tkinter import simpledialog, messagebox, filedialog
import cv2

# Inicializar o ambiente OpenCL
if cv2.ocl.haveOpenCL(): cv2.ocl.setUseOpenCL(True)

# =============== CONTROLE DE IDENTIFICADORES ===============================
#identificadores padrões dos robôs
class ID_Robots:
    '''
        Identificador dos robôs da competição. Que são 3 no total;
        #### Equipe aliada
        * ROBOT_ALLY_GOAL: Goleiro aliado
        * ROBOT_ALLY_1: Primeiro atacante aliado
        * ROBOT_ALLY_2: Segundo atacante aliado
        #### Equipe inimigo
        * ROBOT_ENEMY_GOAL: Goleiro inimigo
        * ROBOT_ENEMY_1: Primeiro atacante inimigo
        * ROBOT_ENEMY_2: Segundo atacante inimigo
    '''
    ROBOT_ALLY_GOAL:int = 0
    ROBOT_ALLY_1:int = 1
    ROBOT_ALLY_2:int = 2

    ROBOT_ENEMY_GOAL:int = 0
    ROBOT_ENEMY_1:int = 1
    ROBOT_ENEMY_2:int = 2


#Identificadores padrões para os pivots
class ID_Pivots:
    ''' Identificador dos pivots
        * CENTER
        * PA1, PA2, PA3 -> Pontos de referÊncia do lado aliado
        * PE1, PE2, PE3 -> Pontos de referência do lado inimigo
    '''
    #Centro do campo
    CENTER: int = 0
    
    #Ponto de referência lado aliado
    PA1: int = 1
    PA2: int = 2
    PA3: int = 3

    #Ponto de referência lado inimigo
    PE1: int = 4
    PE2: int = 5
    PE3: int = 6


#identificador dos objetos
class ID_Objects:
    BALL:       int = 0
    FIELD:      int = 1
    ALLIES:     int = 2
    ENEMIES:    int = 3
    
#identificadores padrões para o campo
class ID_Field:
    '''
        Identificador dos campos
        * GOAL_ALLY: Área do gol aliado que contabiliza ponto para o inimigo;
        * GOAL_ENEMY: Área do gol inimigo que contabiliza ponto para o aliado;

        * GOAL_AREA_ALLY: Área onde o robô goleiro aliado fica.
        * GOAL_AREA_ENEMY: Área onde o robÔ goleiro inimigo fica
    '''
    #Identificador da área que contabiliza gol
    GOAL_ALLY: int = 0
    GOAL_ENEMY: int = 1
    
    #Identificador da área dos goleiros
    GOAL_AREA_ALLY: int = 0
    GOAL_AREA_ENEMY: int = 1


#Identificadores dos times
class ID_Team:
    '''
        Identificador dos times
        * TEAM_ALLY: Time aliado
        * TEAM_ENEMY: Time inimigo
    '''
    TEAM_ALLY: int = 0
    TEAM_ENEMY: int = 1


#Identificador de geometria
class GeometryType:
    '''
        Identificador de tipo de geometria: 
        * POINT2D;
        * QUAD (Quadrilátero)
        * CIRCLE
    '''
    POINT2D: int = 0
    QUAD: int = 1
    CIRCLE: int =2 

# PROTOTIPO DE CLASSES BÁSICAS PARA UM SISTEMA DE COLISÃO
#Tipos de objetos para o sistema de colisão
class ObjTypeVision:
    '''
        Informa qual o tipo de objeto para o sistema de visão:
        * BALL: Bola
        * ROBOT: Robô
        * FIELD: Campo
        * PIVOT: Ponto de referência
        * GOALFIELD: Área dos goleiros ou gols
    '''
    BALL: int = 0
    ROBOT: int = 1
    FIELD: int = 2
    PIVOT: int = 3
    GOALFIELD: int = 4

#Tipos de objetos (que se movem, ou que ficam parados):
'''
@GNOMIO:
    STATIC: FIELD, PIVOT, GOALFIELD
    MOVING: ROBOT, BALL

    Objetos estáticos (STATIC) são pontos fixos no processo
    que não irão "colidir entre si". Objetos que se movem
    (MOVING) são aqueles que se movem na imagem e podem colidir
    tanto com outros objetos MOVING como objetos STATICS. 
    Exemplo:
        Ball = colide com o campo
'''
class ObjTypeMove:
    '''
    Define os tipos de objetos que serão computados
    se são STATIC (estáticos) ou MOVING( móveis). Objetos estáticos não colidem entre si
    mas objetos MOVING colidem entre si e com os estáticos tbm.
    '''
    STATIC: int = 0
    MOVING: int = 1

#identificador do tipo de GPU
class GPUType:
    '''
        Identificador do tipo de GPU:
        * NVidia;
        * AMD;
        * DONTHAVE
    '''
    NVidia: int = 0
    AMD: int = 1
    DONTHAVE: int = -1


#identificador de serviço cuda para o emulador utilizar
class CUDADevice:
    '''
    Estrutura para representar o serviço cuda da máquina
    '''
    #Gerando o objeto de serviço CUDA
    def __init__(self, GPU:GPUType,Version:str):
        self.GPU = GPU
        self.version = Version

# =============== VARIAVEIS GLOBAIS DE EMULAÇÃO ===============================

#Modo de Emulação do aplicativo

MODE_DEFAULT:int = 0
''' Constante de emulação: Modo padrão '''
MODE_USB_CAM: int= 1
''' Constante de emulação:  Modo de emulação através de uma câmera USB ou interna'''
MODE_VIDEO_CAM:int = 3
''' Constante de emulação:  Modo de emulação por meio de um vídeo'''
MODE_IMAGE:int = 2
''' Constante de emulação:  Modo de emulação por meio de uma imagem'''


MODE_CONTROL_ROBOT: int = 4
''' Constante de emulação:  Emulador sendo utilizado na janela de controle'''

#Modos de execução da janela de controle
class ModeControlW:
    '''
        Modos de execução da janela de controle individual dos robôs
    '''
    MANUAL: int = 1
    POINTER: int = 2
    DEFAULT: int = 1
    
# ================== CONTROLE DE ESTRUTURA DE DADOS ========================
#Configurações da Emulação que serão inviadas para o sistema de visão realizar os cálculos

class EConfig:
    '''
        É uma estrutura com as informações passadas pelo emulador ao sistema de visão
        o sistema de visão irá pegar essas informações e se configurar da forma necessária
        terão as variáveis necessárias
    '''
    def __init__(self, offSetWindow =10, offSetErode = 0 ,dimMatrix = 25, Trashhold = 235
                 ,FieldWidth = 0, FieldHeight=0, allyColor=[0,0,0], enemyColor=[0,0,0],ballColor = [0,0,0]
                 , goalAllyColor1=[0,0,0], goalAllyColor2=[0,0,0], atk1AllyColor1=[0,0,0],atk1AllyColor2=[0,0,0], atk2AllyColor1=[0,0,0]
                 , atk2AllyColor2=[0,0,0], emulatorMode = MODE_IMAGE, timer=None,ip_send = "127.0.0.1", port_send = 0, ip_receive = "127.0.0.1", port_receive = 0):
        '''
            Essas são as variáveis base que o sistema de visão utiliza para realizar seu processamento
            são elas as cores dos times, e offsets do cálculo
        '''
        #setando variáveis de configuração do sistema de visão
        self.offSetWindow       = offSetWindow          # valor mínimo da borda da janela
        self.offSetErode        = offSetErode           # quantidade mínima de erosão
        self.dimMatrix          = dimMatrix             # dimensão da matrix de convolução
        self.Trashhold          = Trashhold             # limiar de binarização do sistema
        
        self.fieldWidth         = FieldWidth            # comprimento do campo
        self.fieldHeight        = FieldHeight           # largura do campo
        self.allyColor          = allyColor             # Cor principal do time
        self.enemyColor         = enemyColor            # Cor principal dos inimigos

        self.ballColor          = ballColor             # cor da bola
        self.goalAllyColor1     = goalAllyColor1        # cor 1 do goleiro aliado
        self.goalAllyColor2     = goalAllyColor2        # cor 2 do goleiro aliado
        self.atk1AllyColor1     = atk1AllyColor1        # cor 1 do atacante 1
        self.atk1AllyColor2     = atk1AllyColor2        # cor 2 do atacante 1
        self.atk2AllyColor1     = atk2AllyColor1        # cor 1 do atacante 2
        self.atk2AllyColor2     = atk2AllyColor2        # cor 2 do atacante 2

        self.emulatorMode       = emulatorMode          # modo da emulação
        self.timer              = timer                 # objeto timer

        self.ip_send            = ip_send               # IP de envio protobuff
        self.port_send          = port_send             # Porta de envio protobuff
        self.ip_receive         = ip_receive            # IP de recebimento protobuff
        self.port_receive       = port_receive          # Porta de recebimento protobuff

    #métodos para setar uma variável não precisando ser na inicialização do objeto
    def setOffSetValues(self, ofsWindow, ofsErode, ofsMatrix, ofsTrashhold):
        self.offSetWindow       = ofsWindow          # valor mínimo da borda da janela
        self.offSetErode        = ofsErode           # quantidade mínima de erosão
        self.dimMatrix          = ofsMatrix             # dimensão da matrix de convolução
        self.Trashhold          = ofsTrashhold             # limiar de binarização do sistema

    def setProtobuffConfigs(self, ip_send = "127.0.0.1", port_send = 0, ip_receive = "127.0.0.1", port_receive = 0):
        '''
            Método para setar as configurações de comunicação via protobuff
        '''
        self.ip_send        = ip_send
        self.port_send      = port_send
        self.ip_receive     = ip_receive
        self.port_receive   = port_receive

    #setando as cores pin
    def setMainColors(self, allyColor, EnemyColor, ballColor):
        '''
            Método para cores principais (aliadas e inimigas)
        '''
        self.allyColor = allyColor
        self.enemyColor = EnemyColor
        self.bollColor = ballColor

    #setando as cores individuais
    def setAllyColors(self, g1c1,g1c2, a1c1,a1c2,a2c1,a2c2):
        '''
            Setando as cores principais dos robôs do sistema
        '''
        self.goalAllyColor1     = g1c1        # cor 1 do goleiro aliado
        self.goalAllyColor2     = g1c2        # cor 2 do goleiro aliado
        self.atk1AllyColor1     = a1c1        # cor 1 do atacante 1
        self.atk1AllyColor2     = a1c2        # cor 2 do atacante 1
        self.atk2AllyColor1     = a2c1        # cor 1 do atacante 2
        self.atk2AllyColor2     = a2c2        # cor 2 do atacante 2

    
    #setando dimensões do campo
    def setFieldDimensions(self, width, height):
        '''
            Setando dimensões do campo comprimento e largura
        '''
        self.fieldHeight    = height
        self.fieldWidth     = width

    #setando a forma do emulador
    def setEmulatorMode(self, eMod):
        self.emulatorMode = eMod 


    #Deletar este objeto em tempo de execução
    def delete(self):
        '''
            Deletando o objeto de configuração para liberar memória
        '''
        del self

    #função para puxar as configurações
    def setConfigs(self, offSetWindow =10, offSetErode = 0 ,dimMatrix = 25, Trashhold = 235
                 ,FieldWidth = 0, FieldHeight=0, allyColor=[0,0,0], enemyColor=[0,0,0],ballColor = [0,0,0]
                 , goalAllyColor1=[0,0,0], goalAllyColor2=[0,0,0], atk1AllyColor1=[0,0,0],atk1AllyColor2=[0,0,0], atk2AllyColor1=[0,0,0]
                 , atk2AllyColor2=[0,0,0] ):
                #setando variáveis de configuração do sistema de visão
        self.offSetWindow       = offSetWindow          # valor mínimo da borda da janela
        self.offSetErode        = offSetErode           # quantidade mínima de erosão
        self.dimMatrix          = dimMatrix             # dimensão da matrix de convolução
        self.Trashhold          = Trashhold             # limiar de binarização do sistema
        
        self.fieldWidth         = FieldWidth            # comprimento do campo
        self.fieldHeight        = FieldHeight           # largura do campo
        self.allyColor          = allyColor             # Cor principal do time
        self.enemyColor         = enemyColor            # Cor principal dos inimigos

        self.ballColor          = ballColor             # cor da bola
        self.goalAllyColor1     = goalAllyColor1        # cor 1 do goleiro aliado
        self.goalAllyColor2     = goalAllyColor2        # cor 2 do goleiro aliado
        self.atk1AllyColor1     = atk1AllyColor1        # cor 1 do atacante 1
        self.atk1AllyColor2     = atk1AllyColor2        # cor 2 do atacante 1
        self.atk2AllyColor1     = atk2AllyColor1        # cor 1 do atacante 2
        self.atk2AllyColor2     = atk2AllyColor2        # cor 2 do atacante 2



# =============== CONTROLE DE CLASSES ===============================
#Classe auxiliar para configurar os pontos extremos que irão reconhecer o robô, para análise de colisão.
# [ ] Atualizar para modo GPU com Cupy e Numpy
        

# /// CLASSES GEOMÉTRICAS BÁSICAS (utilizando como base a biblioteca numpy)
#Definição de um ponto 2D no sistema
class Point2D:
    ''' Classe para representar um ponto de 2 dimensões'''
    def __init__(self, px,py):
        self.px = px        #Coordenada x 
        self.py = py        #Coordenada y

        #ponto no formato array do numpy
        self.pos = np.array([px,py])

    #retornando posição central
    def getPos(self):
        '''
        Retorna posição do ponto (x,y)
        '''
        return self.pos 
    
    #Definindo operações com Point2D
    #definindo a soma (x,y)+(a,b) = (x+a, y+b)
    def __add__(self, other):
        if isinstance(other, Point2D):
            return Point2D(self.px +other.px, self.py+other.py)
        elif isinstance(other, tuple):
            return Point2D(self.px +other[0], self.py+other[1])
        else:
            raise TypeError("Operação inválida")
        
    #definindo a subtração de dois pontos (x,y)-(a,b) = (x-a,y-b)
    def __sub__(self, other):
        if isinstance(other, Point2D):
            return Point2D(self.px -other.px, self.py-other.py)
        elif isinstance(other, tuple):
            return Point2D(self.px -other[0], self.py-other[1])
        else:
            raise TypeError("Operação inválida")
        
    #definindo multiplicação entre esses dois pontos 2D
    def __mul__(self, other):
        #Multiplicação por escalar (x,y)*k = (kx,ky)
        if isinstance(other, (int, float)):
            # Se 'other' for um escalar, realizar multiplicação por escalar
            return Point2D(self.px * other, self.py * other)
        
        #multiplicação por uma instância (x,y) * (a,b) = (x*a,y*b) => Necessário criar uma lógica
        elif isinstance(other, Point2D): 
            # Se 'other' for um vetor, realizar produto escalar
            return Point2D(self.px * other.px, self.py * other.py)
        else:
            # Caso contrário, lançar uma exceção ou retornar None
            raise TypeError("Operação de multiplicação não suportada para o tipo de objeto passado.")
    
    # Define o comportamento do operador de string
    def __str__(self):
        return f"Point2D({self.px}, {self.py})"
    
    #define a operação de equalidade
    def __eq__(self, other):
        #Multiplicação por escalar (x,y)*k = (kx,ky)
        if isinstance(other, Point2D):
            if self.px == other.px and self.py == other.py:
                return True 
            else:
                return False 
        elif isinstance(other, tuple):
            try: 
                if self.px == other[0] and self.py == other[0]:
                    return True 
                else:
                    return False
            except:
                return False 
        else:
            # Caso contrário, lançar uma exceção ou retornar None
            raise TypeError("Não é possível tomar a igualdade entre dois valores diferentes")
    
    #define o tamanho do objeto
    def __len__(self):
        return 2 
    
    #define como pegar um valor desse método
    def __getitem__(self, index):
        ''' Retorna o valor correspondente ao índice'''
        if index == 0:
            return self.px
        elif index == 1:
            return self.py
        else:
            raise IndexError("Índice fora do intervalo para Point2D")
#Definição de um Quadrilátero 
class Quad:
    '''
        Definição de um Quadrilátero para o código.
    '''
    def __init__(self, P1:Point2D, P2:Point2D, P3:Point2D, P4:Point2D):
        ''' Necessário informar 4 pontos para ele interpretar e juntar'''
        self.p1 = P1                #Ponto extremo 1
        self.p2 = P2                #Ponto extremo 2
        self.p3 = P3                #Ponto extremo 3
        self.p4 = P4                #Ponto extremo 4

        #Pontos no formato array do numpy
        self.points = np.array([P1,P2,P3,P4])

    #puxar os pontos
    def getPoint(self):
        '''
            Retorna os pontos associados a esse Quadrilátero num array
        '''
        return self.points
     
#Definição de um círculo
class Circle(GeometryType):
    '''
        Definição de uma estrutura para representar um círculo
    '''
    def __init__(self,Center:Point2D, radius:int):
        self.center = Center
        self.radius = radius

    #retornando raio do círculo
    def getRadius(self):
        '''
            Retorna o raio do círculo
        '''
        return self.center
    
    #retornando ponto central
    def getCenter(self):
        '''
            Retorna o centro do círculo
        '''
        return self.center.getPos()


#Definindo um polígono
'''...'''

#========================= /// CLASSES SEMÂNTICAS DO CÓDIGO // ===================
#Classe responsável por guardar as informações dos pontos de apoio
class Pivot:
    '''
    Representa um ponto com identificador, ou seja, um ponto importante no jogo.
    '''
    def __init__(self, id:ID_Pivots, Point:Point2D = Point2D(0,0)):
        self.id = id                #identificador
        self.posX = Point.px        #posição x
        self.posY = Point.py        #posição y
        self.pos = Point.pos        #formato array do numpy

        #Informações internas do tipo de objeto
        self.ObjType = ObjTypeMove.STATIC
        self.objTypeSystem = ObjTypeVision.PIVOT

    #Atualizar a posição do ponto de referência
    def updatePos(self,px,py):
        '''
            Atualiza a posição do ponto de referência
        ''' 
        self.posX=px 
        self.posY=py

#Objeto para representar a borda de um robô que será utilizada para análisar colisão
class BorderBox:

    def __init__(self, type:GeometryType, Infos):
        '''
        Essa classe representa a borda de colisão do objeto, podendo ser de três tipos:
        * Círculo (GeometryType.CIRCLE)
        * Quadrilátero (GeometryType.QUAD)
        * Ponto ((GeometryType.POINT2D)

        Cada tipo de borderbox terá um sistema de colisão diferente.
        '''
        #Adquire o tipo do bbox
        self.type = type

        #Informa como o objeto deve guardar suas informações
        if(self.type == GeometryType.CIRCLE): #verifica se é um círculo
            self.Center = Infos.center
            self.radius = Infos.radius

        elif(self.type == GeometryType.QUAD): #verifica se é um retângulo
            self.p1 = Infos.p1
            self.p2 = Infos.p2
            self.p3 = Infos.p3
            self.p4 = Infos.p4

        elif(self.type == GeometryType.POINT2D): #verifica se é um ponto 2D
            self.px = Infos.px
            self.py = Infos.py

        else:
            raise TypeError("Está tentanto atribuir um objeto que não é um tipo geométrico válido")
    #atualizando informações do Borderbox
    def attPosition(self, newObj):
        #Informa como o objeto deve guardar suas informações
        if(self.type == GeometryType.CIRCLE and isinstance(newObj,Circle)): #verifica se é um círculo
            self.Center = newObj.center
            self.radius = newObj.radius 

        elif(self.type == GeometryType.QUAD and isinstance(newObj,Quad)): #verifica se é um retângulo
            #Falta criar essa lógica
            self.p1 = newObj.p1 
            self.p2 = newObj.p2 
            self.p3 = newObj.p3 
            self.p4 = newObj.p4 

        elif(self.type == GeometryType.POINT2D and isinstance(newObj,Point2D)): #verifica se é um ponto 2D
            self.px = newObj[0]
            self.py = newObj[1]

        else:
            raise TypeError("Esta tentando atribuir um objeto que não é um tipo geométrico válido")

#Classe responsável por organizar as áreas no campo
class AreaField:
    '''
        A classe representa uma área importante do campo, podendo ser a área
        do goleiro aliado ou inimigo, bem como  o próprio campo e o campo aliado e inimigo.
    '''
    def __init__(self, id:ID_Field,rect:Quad = Quad(Point2D(0,0),Point2D(0,0),Point2D(0,0),Point2D(0,0))):
        #Setando identificador da área
        self.Id = id
        self.rect = rect

        #Gerando uma bbox para sistema de colisões
        self.ObjType = ObjTypeMove.STATIC
        self.objTypeSystem = ObjTypeVision.GOALFIELD

        #configurando a border box da área
        self.bbox = BorderBox(GeometryType.QUAD,rect)

    #Setando o retângulo que o representa
    def setRect(self,rect:Quad):
        '''
            Informo qual o retângulo que será representado por essa área
        '''
        self.rect = rect

    def setId(self, id:ID_Field):
        '''
            Informo qual é o identificador novo dessa área.
        '''
        self.Id = id

#definindo classe de view para o robô, que irá armazenar a posição de uma
#janela que informa onde o robô estará, para reduzir o processamento
class ViewBot:
    '''
        Classe de view que representa uma janela da imagem onde o robô se encontra.
    '''
    def __init__(self, Position:Point2D, DimMatrix: int):
        '''
        Classe de view que representa uma janela da imagem onde o robô se encontra.
        '''
        #Centro (x,y)
        self.center = Position.pos

        #passo
        self.DimMatrix = DimMatrix
        self.step = self.DimMatrix/2.0

        #Encontrando pontos
        self.Pe1 = self.center + np.array([-1,-1])*self.step
        self.Pe2 = self.center + np.array([1,-1])*self.step
        self.Pe3 = self.center + np.array([1,1])*self.step
        self.Pe4 = self.center + np.array([-1,1])*self.step

        #Gerando retângulo para guardar as informações
        self.rect = Quad(self.Pe1, self.Pe2, self.Pe3, self.Pe4)

    #atualizando a posição da ViewBot pela posição central
    def updateViewBot(self, newPosition: Point2D):
        '''
            Atualiza a viewBot com base na nova posição programada (xf, yf)
        '''
        #Centro (x,y)
        self.center = newPosition.pos
        #Encontrando pontos
        self.Pe1 = self.center + np.array([-1,-1])*self.step
        self.Pe2 = self.center + np.array([1,-1])*self.step
        self.Pe3 = self.center + np.array([1,1])*self.step
        self.Pe4 = self.center + np.array([-1,1])*self.step

        #Gerando retângulo para guardar as informações
        self.rect = Quad(self.Pe1, self.Pe2, self.Pe3, self.Pe4)

    #atualizando posição do ViewBot por um passo
    def translateViewBot(self, stepView:Point2D):
        '''
            Função responsável por translatar a view, sendo necessário passar o
            vetor de passo (stepView), que consiste de duas coordenadas (px,py)
        '''
        #Centro (x,y) é passado por um passo (pa, pb) => (x+pa, y+pb)
        self.center = self.center + stepView.pos

        #Encontrando pontos
        self.Pe1 = self.center + np.array([-1,-1])*self.step
        self.Pe2 = self.center + np.array([1,-1])*self.step
        self.Pe3 = self.center + np.array([1,1])*self.step
        self.Pe4 = self.center + np.array([-1,1])*self.step

        #Gerando retângulo para guardar as informações
        self.rect = Quad(self.Pe1, self.Pe2, self.Pe3, self.Pe4)
    
    #retornando os pontos que estão guardados na variável retângulo
    def getPoints(self):
        '''
            Retorna os pontos que caracterizam a view
        '''
        return self.rect.points
    
    #setando nova dimensão do viewBot
    def setDimension(self, newDimension):
        #passo
        self.DimMatrix = newDimension
        self.step = self.DimMatrix/2.0

        #Encontrando pontos
        self.Pe1 = self.center + np.array([-1,-1])*self.step
        self.Pe2 = self.center + np.array([1,-1])*self.step
        self.Pe3 = self.center + np.array([1,1])*self.step
        self.Pe4 = self.center + np.array([-1,1])*self.step

#========================| Gerando classe Timer | ==============================

#configurando objeto timer de alta precisão para pegar o passar do tempo de processamento
class HighPrecisionTimer:
    '''
        Classe para representar um timer de alta precisão que será utilizado
        no programa para recuperar informações tempo
    '''
    def __init__(self, master):
        self.master = master
        self.start_time = None
        self.elapsed_time = 0
        self._isRunning = False
        
    def run(self):
        '''
            Inicializar a contagem do timer
        '''
        self._isRunning = True
        self.start_time = time.time()
    
    def stop(self):
        '''
            Parar a contagem do timer.
        '''
        if self.start_time is not None:
            current_time = time.time()
            self.elapsed_time += (current_time - self.start_time)*1000  # Multiplica por 1000 para obter milissegundos
            self.start_time = None
            self._isRunning = False
        else:
            print("O timer ainda não foi iniciado...")

    def reset(self):
        '''
            Resetar o timer para 0.
        '''
        self.start_time = None
        self.elapsed_time = 0
        self._isRunning = False
        
    def getElapsedTime(self):
        '''
            Retorna o tempo que se passou desde que foi ligado, sem considerar o tempo 
            que passou pausado pela função stop()
        '''
        if self.start_time is not None:
            current_time = time.time()
            elapsed_ms = (self.elapsed_time + (current_time - self.start_time)*1000)
            return elapsed_ms
        else:
            return self.elapsed_time
    
    def isRunning(self):
        '''
            Retorna se o timer está ou não parado.
        '''
        return self._isRunning

class Benchmark:
    def __init__(self):
        self.start_times = {}
        self.times = defaultdict(list)
    
    def tic(self):
        self.start = time.perf_counter()
        
    def toc(self, label="Tempo"):
        elapsed = (time.perf_counter() - self.start) * 1000
        self.times[label].append(elapsed)
        # Removido print para não atrasar processamento
        return elapsed
    
    def get_last(self, label):
        if label in self.times and self.times[label]:
            return self.times[label][-1]
        return 0.0
    
    def get_avg(self, label):
        if label in self.times and self.times[label]:
            return sum(self.times[label]) / len(self.times[label])
        return 0.0
    
    def reset(self):
        self.times.clear()
    
#========================= /// CLASSE BÁSICA DE EXECUÇÃO // =====================
'''
 @GNOMIO: Essa estrutura deveria representar de forma simples a forma de captura de imagens, sendo elas tanto por câmera, ou por arquivos. E funcionará de forma a simplificar a parte semâtica do código, contudo, ainda está em fase de estruturar
'''
# Representa o hardware ou software de captura de imagens
class CaptureMode:
    '''
        Modos de captura da câmera: DEFAULT, CAM, IMG, VIDEO.
    '''
    DEFAULT: int = 0
    CAM: int = 1
    DEFAULT = CAM
    IMG: int = 2
    VIDEO: int = 3

#enumeração para o modo de focalização da câmera
class FocusMode:
    '''
        Representar qual o modo de foco da câmera, se é automático ou manual
    '''
    AUTO: int = 0
    MANUAL: int =1 



class Capture:
    def __init__(self, mode: CaptureMode.DEFAULT, useGpu: bool = False):
        self.mode = mode
        self.image = None
        self.isCamRunning = False
        self._hasGPU = useGpu
        self.cuda = None
        self.idCam = 0
        self.modeCam: FocusMode = FocusMode.AUTO
        self.focusManual = 155
        self._camHasFocusControl = False
        self._hasCamera = False
        self.imgPath = None
        self.videoPath = None
        self.CAM = None

        # --- VARIÁVEIS DE THREAD ---
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        # ---------------------------

        self.GPUMode(self._hasGPU)

    def setMode(self, mode: CaptureMode):
        self.mode = mode

    def setModeFocus(self, mode: FocusMode = FocusMode.AUTO):
        if self.mode == CaptureMode.CAM and (self.CAM is not None):
            if mode == FocusMode.AUTO:
                if not self.CAM.set(cv2.CAP_PROP_AUTOFOCUS, 0):
                    self.modeCam = FocusMode.AUTO
                    self._camHasFocusControl = False
                    return False
                else:
                    self.modeCam = mode
                    self._camHasFocusControl = True
                    return True
            elif mode == FocusMode.MANUAL:
                if not self.CAM.set(cv2.CAP_PROP_FOCUS, self.focusManual):
                    self._camHasFocusControl = False
                    return False
                else:
                    self.modeCam = mode
                    self._camHasFocusControl = True
                    return True
            else:
                self._camHasFocusControl = False
                self.modeCam = FocusMode.AUTO
                return False 
        else:   
            self._camHasFocusControl = False
            self.modeCam = FocusMode.AUTO
            return False

    def setFocusManual(self, value: float):
        if self.mode == CaptureMode.CAM and self._camHasFocusControl:
            if self.modeCam == FocusMode.MANUAL and (self.CAM is not None):
                self.focusManual = np.clip(value, 0, 255)
                self.CAM.set(cv2.CAP_PROP_FOCUS, self.focusManual)

    def GPUMode(self, useGpu: bool):
        self._hasGPU = useGpu
        if self._hasGPU:
            self.cuda = cv2.cuda.GpuMat()
            if self.cuda is not None:
                print("[CAPTURA]: Construído com GPU / CUDA")
        else:
            self.cuda = None

    def setIdCam(self, id):
        self.idCam = int(id)
        if self.CAM is not None:
            self.running = False
            if self.thread is not None:
                self.thread.join(timeout=1.0)
            self.CAM.release()

        if self.mode == CaptureMode.CAM:
            try:
                self.CAM = cv2.VideoCapture(self.idCam)
                if not self.CAM.isOpened():
                    return False
                else:
                    self._hasCamera = True
                    self.CAM.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    self.running = True
                    self.thread = threading.Thread(target=self._update_cam_loop, daemon=True)
                    self.thread.start()
                    return True
            except:
                return False
        return False

    def _update_cam_loop(self):
        """ Thread loop para limpar buffer da câmera """
        while self.running and self.CAM is not None and self.CAM.isOpened():
            ret, frame = self.CAM.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def setImagePath(self, pathImg):
        self.imgPath = pathImg

    def setVideoPath(self, pathVideo):
        self.videoPath = pathVideo
        if self.mode == CaptureMode.VIDEO:
            if self.CAM is not None:
                self.CAM.release()
            self.CAM = cv2.VideoCapture(self.videoPath)
            if not self.CAM.isOpened():
                print("[Capture] Erro ao abrir vídeo:", self.videoPath)

    def getImageNoCuda(self):
        if not self._hasGPU:
            if self.mode == CaptureMode.IMG:
                if self.imgPath:
                    self.image = cv2.imread(self.imgPath)
                return self.image
            
            elif self.mode == CaptureMode.CAM:
                with self.lock:
                    if self.latest_frame is not None:
                        self.image = self.latest_frame.copy()
                return self.image

            elif self.mode == CaptureMode.VIDEO:
                if self.CAM is not None and self.CAM.isOpened():
                    ret, frame = self.CAM.read()
                    if ret:
                        self.image = frame
                        return self.image
                    else:
                        self.image = None
                        return None # Fim do vídeo alcançado
                return None
        return None

    def getImageCuda(self):
        if self._hasGPU:
            if self.mode == CaptureMode.IMG:
                if self.imgPath:
                    self.image = cv2.imread(self.imgPath)
                    if self.image is not None:
                        self.cuda.upload(self.image)
                        return self.cuda

            elif self.mode == CaptureMode.CAM:
                with self.lock:
                    if self.latest_frame is not None:
                        self.image = self.latest_frame.copy()
                
                if self.image is not None:
                    self.cuda.upload(self.image)
                    return self.cuda

            elif self.mode == CaptureMode.VIDEO:
                if self.CAM is not None and self.CAM.isOpened():
                    ret, frame = self.CAM.read()
                    if ret:
                        self.image = frame
                        self.cuda.upload(self.image)
                        return self.cuda
                    else:
                        self.image = None
                        return None # Fim do vídeo alcançado
        return None

    def getImage(self):
        if self._hasGPU:
            return self.getImageCuda()
        else:
            return self.getImageNoCuda()

    def reset(self):
        # 1. Para Thread de Câmera
        self.running = False
        if hasattr(self, 'thread') and self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

        # 2. Reseta Variáveis de Estado
        self.latest_frame = None
        self.mode = CaptureMode.DEFAULT
        self.image = None
        self.isCamRunning = False
        self.idCam = 0 
        self.imgPath = None
        self.videoPath = None
        self._hasGPU = False
        self._hasCamera = False
        self._camHasFocusControl = False

        # 3. Libera Recursos do OpenCV e CUDA
        if self.CAM is not None:
            self.CAM.release()
            self.CAM = None

        if self.cuda is not None:
            self.cuda.release()

    def __del__(self):
        self.reset()
        
#classe para gerenciar a thread de captura de dados
class CameraCaptureThread(threading.Thread):
    '''
        Essa classe é responsável por gerar a Thread que irá capturar imagens
        e salvar elas num deque, que será acessado pelo emulador.
    '''
    def __init__(self, main, settingMenu,capture_instance: Capture, deque:deque, interval=0.016):
        super().__init__()
        self.capture_instance = capture_instance
        self.interval = interval
        self._is_running = False
        self._main = main
        self.deque = deque
        self.daemon = True
        self.menu = settingMenu

        self.latest_frame = None # Variável que guardará sempre o frame mais novo
        self.lock = threading.Lock() # Segurança para não ler enquanto escreve
        self.running = False
        self.thread = None


    def run(self):
        self._is_running = True
        while self._is_running:
            #procura imagem
            if self.capture_instance.mode == FocusMode.AUTO:
                #print("FOCO AUTOMÁTICO")
                new_image = self.capture_instance.getImage()
                if new_image is None:
                    pass
                else:
                    self.deque.append(new_image)  # Enviando a nova imagem para a fila
                #print(self.deque[-1])
                time.sleep(self.interval)
            else: #modo manual

                self.capture_instance.setFocusManual(self._main.FocusValue)
                new_image = self.capture_instance.getImage()
                #print(self.capture_instance.modeCam, ' :', self.capture_instance.focusManual)
                if new_image is None:
                    pass
                else:
                    self.deque.append(new_image)  # Enviando a nova imagem para a fila
                #print(self.deque[-1])
                time.sleep(self.interval)
    
    def stop(self):
        #liberar recursos
        if self.capture_instance.mode == CaptureMode.CAM:
            self.capture_instance.CAM.release()
        print("Recursos liberados")
        #parando
        self._is_running = False


#==================================== CLASSES PARA INTERFACE =========================
#definindo função para indicar estado do controlador
class StateSquare(Frame):
    '''
        É apenas uma classe para representar um quadrado na interface.
    '''
    def __init__(self, master=None, variable=None, btn= None):
        super().__init__(master, bg="white")
        self.square_size = 10
        self.status = variable
        self.square = Canvas(self, width=self.square_size, height=self.square_size, bd=1, relief="solid", bg="red")
        self.square.grid(row=0, column=0, padx=5, pady=5)
        self.text_var = StringVar()
        self.text_var.set("Parado")
        self.text_label = Label(self, textvariable=self.text_var, bg="white")
        self.text_label.grid(row=0, column=1, padx=5, pady=5)
        self.button = btn

    #modificar estado
    def toggle_status(self):
        if self.status == True:
            self.set_status(False)
        else:
            self.set_status(True)


    #definir estado
    def set_status(self, status):
        self.status = status
        if self.status == True:
            self.square.config(bg="green")
            self.text_var.set("Em execução")
            self.button.config(text="Parar o processamento")
        elif self.status  == False:
            self.square.config(bg="red")
            self.text_var.set("Parado")
            self.button.config(text="Iniciar processamento")
