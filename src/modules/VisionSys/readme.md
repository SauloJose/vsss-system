# Sistema de Visão VSS (Vision System Soccer) v3.3.22

## 📋 Visão Geral

O **Sistema de Visão VSS** é um módulo de detecção e rastreamento de objetos em tempo real, projetado para competições de futebol de robôs. Ele processa imagens do campo, identifica a bola, robôs aliados e inimigos, utilizando técnicas de visão computacional combinadas com **filtros de Kalman** (EKF para robôs) para oferecer rastreamento robusto e preditivo.

A arquitetura atual foi reorganizada em **9 blocos funcionais** que separam claramente:

1. Métodos de controle (inicialização, configuração, reset)
2. Funções auxiliares de processamento de imagem
3. Desenho e depuração
4. Transformações geométricas e homografia
5. Localização geométrica por tags (protocolo de cores)
6. Detecção principal (campo, bola, jogadores)
7. Detecção adaptativa com ROI (Kalman)
8. Predição e fallback com Kalman
9. Pipelines principais (detecção completa e rastreamento filtrado)

**Versão atual:** v3.3.22 (BETA)  
**Última modificação:** 08/08/206 (nota: data fictícia)  
**Autor:** Saulo (update)

---

## 🎯 Características Principais

- **Detecção em tempo real** de múltiplos objetos (6 robôs + bola)
- **Rastreamento por Filtro de Kalman**:
  - **EKF** para robôs com modelo diferencial (velocidades das rodas no estado)
  - **KF linear** para bola (posição + velocidade)
- **ROI adaptativa** baseada na covariância do filtro para busca otimizada
- **Sistema hierárquico de cores** (`TreeColors`) com tolerância a variações de iluminação
- **Pipeline híbrido**: detecção completa nos primeiros frames, depois rastreamento filtrado
- **Fallback inteligente**: predição do Kalman mantém a estimativa durante oclusões
- **Tratamento de colisões**: detecção de blobs com múltiplos robôs e separação por erosão iterativa
- **Gating de distância**: rejeição de medições implausíveis (saltos > `MAX_JUMP_CM`)
- **Watchdog** para reset automático do filtro após perda prolongada
- **Suporte a GPU** (CUDA) para processamento paralelo
- **Modo debug** com visualização de ROIs, máscaras binárias e vetores de direção

---

## 🏗️ Arquitetura da Classe `VisionSystem`

A classe principal está organizada nos seguintes blocos funcionais (ordem de declaração no código):

| Bloco | Nome | Responsabilidade |
|-------|------|------------------|
| 1 | **Métodos de controle** | Inicialização, configuração, criação de objetos, reset, obtenção de dados |
| 2 | **Funções auxiliares** | Processamento de imagem (gray, blur, binarização, morfologia), redução de janela, conversão de cores, manipulação de contornos |
| 3 | **Funções de desenho e debug** | Renderização de círculos, setas, textos, máscaras e ROIs para depuração |
| 4 | **Processamento geométrico puro** | Cálculo de homografia, transformação de coordenadas (real ↔ virtual ↔ cm), detecção de formas (quadrados) |
| 5 | **Localização geométrica do robô** | Implementação do protocolo de tags (Tag_Time, T1, T2) para orientação e identificação por cores |
| 6 | **Detecção principal** | `DetectField`, `DetectBall`, `DetectPlayers` – pipeline completo de detecção por frame (modularizada em 3 partes) |
| 7 | **Detecção adaptativa com ROI** | `SearchBots`, `DetectBotInRoi`, `SearchBot`, `SearchBall` – busca em janelas preditas pelo Kalman |
| 8 | **Predição e filtragem Kalman** | Cálculo de ROI, predição de posição, fallback para objetos perdidos |
| 9 | **Pipelines principais** | `Proc` (detecção completa), `ProcessImg` (orquestrador), `FilteredDetection` (rastreamento otimizado) |

---

## 🔧 Fluxo de Processamento

### 1. Modo Imagem (único quadro)
- `ProcessImg` chama `Proc` com `force_field_detect=True`.
- `Proc` executa detecção completa: campo → bola → jogadores.
- Resultado é retornado sem rastreamento.

### 2. Modo Vídeo (contínuo, com Kalman)
- **Fase de aquecimento** (primeiros 12000 frames ≈ 400s a 30fps):
  - Executa `Proc` com `force_field_detect=False` (reutiliza ROI do campo).
  - Alimenta os filtros de Kalman com as medições.
- **Fase de rastreamento** (após aquecimento):
  - `ProcessImg` chama `FilteredDetection`:
    1. Prediz ROI da bola e dos robôs usando `PredictBall`/`PredictRobot`.
    2. Busca apenas dentro dessas janelas (redução de ~80% da área).
    3. Se encontrado, atualiza o filtro com `updatePosition`.
    4. Se não encontrado, incrementa contador de falhas (`missed_frames`).
    5. Se exceder limite (`MAX_MISSED_FRAMES_*`), marca para reset do filtro.
    6. Fallback: usa predição do Kalman para manter estimativa sem atualizar.
- **Recalibração periódica**: a cada `newProcTime` (10s), executa `Proc` para corrigir deriva.

### 3. Gerenciamento de Falhas (Watchdog)
- **Campo**: se `DetectField` falhar por 10 quadros consecutivos, reset completo do sistema.
- **Bola**: se não detectada por 10 quadros, o filtro é reiniciado na próxima detecção.
- **Robôs**: se não detectados por 15 quadros, o filtro é reiniciado na próxima detecção.

---

## 🎨 Sistema de Cores – `TreeColors`

A classe `TreeColors` gerencia a identificação hierárquica de robôs com base em três cores:

- **Cor do time** (Tag_Time) – presente na parte inferior do robô (retângulo `L × L/2`).
- **Cor primária** (Tag_T1) – canto superior esquerdo (quadrado `L/2 × L/2`).
- **Cor secundária** (Tag_T2) – canto superior direito (quadrado `L/2 × L/2`).

Onde `L = 7.5 cm` (dimensão padrão do robô).

### Cadastro
```python
tree.add_robot(team_id, robot_id, main_hsv, primary_hsv, secondary_hsv)
```

### Busca
```python
match = tree.find_by_colors(main_candidate, primary_candidate, secondary_candidate)
# Retorna: {'team': team_id, 'robot_id': robot_id} ou None
```

### Tolerâncias
- **Hue**: tratamento de wrap circular (0–179) com tolerância ajustável (padrão ±10).
- **Saturação e Valor**: limites configuráveis (padrão ±50).

---

## 📐 Transformações Geométricas

O sistema utiliza **homografia** para mapear a imagem real para uma imagem virtual de dimensões fixas (645×413 px, proporção 3 px/cm). As coordenadas são então convertidas para um sistema O' com origem no centro do campo (em cm).

| Método | Entrada | Saída | Descrição |
|--------|---------|-------|-----------|
| `GetHomographyMatrix` | 4 pontos reais, 4 pontos virtuais | Matriz 3×3 | Calcula a homografia. |
| `TransformPoint` | ponto (px real) | ponto (px virtual) | Aplica homografia direta. |
| `InvTransformPoint` | ponto (px virtual) | ponto (px real) | Aplica homografia inversa. |
| `GetPointVirtual` | ponto (px virtual) | (x, y) em cm no sistema O' | Converte para coordenadas do campo (centro em (0,0)). |
| `GetImageIndice` | (x, y) em cm | ponto (px virtual) | Inverso de `GetPointVirtual`. |
| `GetImageRealIndice` | (x, y) em cm | ponto (px real) | Combina `GetImageIndice` + `InvTransformPoint`. |

---

## 🤖 Filtro de Kalman – Robôs (EKF)

**Modelo de estado** (6 dimensões):
```
[x, y, θ, vL, vR, ω]^T
```

**Modelo de movimento** (diferencial):
```
v = (vL + vR) / 2
x' = x + v·dt·cos(θ)
y' = y + v·dt·sin(θ)
θ' = θ + ω·dt
vL' = vL
vR' = vR
ω'  = ω
```

**Observação**: `[x, y, θ]` (posição e orientação).

**ROI adaptativa**: baseada na covariância predita, dimensionada por `scale_std=3`. A ROI é calculada em centímetros e convertida para pixels via `GetRoiImg`.

---

## ⚽ Filtro de Kalman – Bola

**Modelo de estado** (5 dimensões):
```
[x, y, θ, vx, vy]^T
```

**Modelo de movimento** (linear):
```
x' = x + vx·dt
y' = y + vy·dt
θ' = θ   (derivado do movimento, não medido)
vx' = vx
vy' = vy
```

**Observação**: `[x, y]` (apenas posição). A orientação θ é derivada do vetor velocidade e não é observada diretamente.

---

## 🔍 Detecção Filtrada – `FilteredDetection`

Pipeline otimizado que substitui a detecção completa após o aquecimento:

1. **Recorte do campo** a partir do ROI salvo (`viewCapture.cooVetor`).
2. **Predição da bola** → `PredictBall` → ROI em pixels.
3. **Busca da bola** na ROI → `SearchBall`.
   - Se encontrada: atualiza Kalman com `updatePosition`.
   - Se não: incrementa `missed_frames_ball`; se > limite, marca para reset.
4. **Predição dos robôs** → `PredictRobot` para cada robô (aliados e inimigos).
5. **Busca de robôs** na ROI → `SearchBots` → chama `DetectBotInRoi` para cada um.
   - Para cada robô encontrado: atualiza Kalman com `updatePosition`.
   - Para cada robô não encontrado: `HandleRobotLoss` (predição ou reset).
6. **Fallback**: se o robô/bola está perdido, usa a predição do Kalman para manter a estimativa (sem atualizar o filtro).
7. **Desenho de debug**: ROIs, círculos, setas e textos.

### Modularização da Detecção de Jogadores

A função `DetectPlayers` foi dividida em três partes reutilizáveis:

- **Parte 1 – `DetectPlayerCandidates`**  
  Gera máscara genérica, extrai contornos e classifica por tamanho. Para blobs grandes (colisões), delega para a Parte 2.

- **Parte 2 – `ResolveCollisionBlob`**  
  Trata blobs com múltiplos robôs usando erosão iterativa (`_SplitBlobByErosion`) e complementa com predição do Kalman (`_GetKalmanPredictedCenters`).

- **Parte 3 – `AssociatePlayerCandidates`**  
  Identifica a qual robô específico cada candidato pertence (por cores e posição), atualiza o filtro de Kalman e gerencia a árvore de cores.

---

## ⚡ Otimizações de Desempenho

- **Cache de HSV**: `hsv_fieldReduce` é calculado uma vez por frame e reutilizado.
- **Estruturas morfológicas pré-criadas**: `struct_ellipse5`, `struct_rect11`, `kernel_morph`.
- **Recorte de campo com cache**: quando `force_field_detect=False`, reutiliza ROI anterior.
- **Busca em janelas reduzidas**: apenas a área prevista pelo Kalman é processada.
- **Uso de `SafeCall`**: tratamento de exceções para evitar crashes e manter o fluxo.
- **Caching de pontos de homografia** para evitar recriação de arrays numpy a cada frame.

---

## 🖥️ Uso Básico

```python
from modules.VisionSys.vision_system import VisionSystem
from modules.VisionSys.components.config import EConfig

# Configuração
config = EConfig()
config.fieldWidth = 150
config.fieldHeight = 130
config.ballColor = [10, 200, 200]  # HSV
config.allyColor = [30, 200, 200]
config.enemyColor = [110, 200, 200]

# Inicialização
vs = VisionSystem(config=config, debug=True)

# Loop de captura
while True:
    img = capture.read()  # imagem BGR
    timestamp = time.time() * 1000  # ms

    # Processamento
    result_img = vs.ProcessImg(img, debug=True)

    # Obter objetos
    objects = vs.GetObjects()
    ball = objects[ID_Objects.BALL]
    allies = objects[ID_Objects.ALLIES]

    if ball.detected:
        print(f"Bola: ({ball.x:.2f}, {ball.y:.2f}) cm")
    else:
        print(f"Bola predita: ({ball.x:.2f}, {ball.y:.2f}) cm")
```

---

## 🧪 Debug e Visualização

Com `debug=True`:
- **Máscaras binárias** (`GetDebugImages`): objetos, bola, jogadores, aliados.
- **Desenho na imagem**: círculos, IDs, setas de direção.
- **ROIs do Kalman**: retângulos amarelos (bola) e coloridos (robôs).
- **Imagem virtual**: campo com posições projetadas e orientações.

Para obter as máscaras:
```python
bin_objects, bin_ball, bin_players, bin_allies = vs.GetDebugImages()
```

---

## 📁 Estrutura de Arquivos (Módulos)

```
modules/VisionSys/
├── vision_system.py          # Classe VisionSystem (este arquivo)
├── components/
│   ├── objects.py            # Definições de ID_Team, ID_Robots, ID_Objects
│   ├── viewcapture.py        # ViewCapture (ROI do campo)
│   ├── field.py              # Classe Field
│   ├── ball.py               # Classe Ball (com Kalman)
│   ├── robot.py              # Classe Robot (com EKF)
│   └── combination.py        # TreeColors e utilitários de cores
└── ...
```

---

## ⚙️ Parâmetros Ajustáveis

| Parâmetro | Valor padrão | Descrição |
|-----------|--------------|-----------|
| `MAX_MISSED_FRAMES_BALL` | 10 | Frames sem detecção antes de resetar o Kalman da bola. |
| `MAX_MISSED_FRAMES_ROBOT` | 15 | Frames sem detecção antes de resetar o Kalman do robô. |
| `MAX_JUMP_CM` | 50.0 | Distância máxima (cm) aceita entre a predição e a medição (gating). |
| `newProcTime` | 10000 ms | Intervalo para recalibração completa (detecção de campo). |
| `scale_std` (no `get_roi`) | 3.0 | Multiplicador do desvio padrão para definir ROI. |
| `hue_tolerance` (cores) | 10 | Tolerância do matiz (circular). |
| `saturation_tolerance` | 50 | Tolerância da saturação. |
| `value_tolerance` | 50 | Tolerância do valor (brilho). |

---

## 🧠 Considerações de Projeto

- **Separação de responsabilidades**: cada bloco funcional tem um propósito claro, facilitando manutenção e testes.
- **Robustez**: tratamento de exceções e fallbacks garantem que o sistema não pare abruptamente.
- **Extensibilidade**: novos robôs ou cores podem ser adicionados via `TreeColors` sem modificar a lógica principal.
- **Performance**: o pipeline híbrido (completo ↔ filtrado) reduz drasticamente o custo computacional após a estabilização do Kalman.
- **Precisão**: o EKF com modelo diferencial captura melhor a dinâmica dos robôs, especialmente em curvas e paradas.
- **Tratamento de colisões**: a abordagem baseada em erosão + predição do Kalman melhora a recuperação em situações de aglomeração.

---

## 🔮 Próximos Passos (Patch Notes v3.3.22)

- Refinamento do tratamento de colisões (separação de blobs com múltiplos robôs).
- Ajuste fino dos parâmetros do filtro de Kalman (sintonização).
- Testes extensivos em condições reais de jogo.

---

Este sistema foi desenvolvido e testado para competições VSS, oferecendo um equilíbrio entre precisão, velocidade e adaptabilidade às condições de jogo.