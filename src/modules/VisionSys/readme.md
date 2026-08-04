Com base nas atualizações do código, aqui está o README.md revisado:

# Sistema de Visão VSS (Vision System Soccer) v2.3.2

## 📋 Visão Geral

O **Sistema de Visão VSS** é um módulo de detecção de objetos em tempo real desenvolvido para competições de futebol de robôs. O sistema processa imagens de campo, detectando bola, robôs aliados e inimigos, utilizando técnicas avançadas de visão computacional e filtros de Kalman para rastreamento.

### 🎯 Características Principais
- Detecção em tempo real de múltiplos objetos
- Suporte a processamento CPU/GPU
- Sistema de coordenadas virtual para mapeamento preciso
- **Filtros de Kalman EKF** para predição de movimento com modelo diferencial
- **Sistema de cores hierárquico** para identificação robusta de robôs
- Arquitetura multi-thread para alto desempenho
- **Detecção filtrada** com ROI baseada em predição do Kalman

## 🏗️ Arquitetura do Sistema

### Classes Principais

### 1. `VisionSystem` - Classe Principal

**Responsabilidade**: Orquestrar todo o pipeline de processamento de visão

**Métodos Principais Atualizados**:

#### `processImg(img, debug)`
Gerencia o modo de operação inteligente:
- **Fase 1** (primeiros 60 frames): Processamento completo para alimentar o Kalman
- **Fase 2**: Detecção filtrada usando predição do Kalman
- **Atualização Periódica**: Processamento completo a cada `newProcTime`

#### `filtered_detection(img, currentTime, debug)`
Nova detecção otimizada:
- Usa Kalman para prever ROIs reduzidas
- Busca seletiva em janelas otimizadas
- Fallback para predição quando detecção falha

#### `search_bots(img, timestamp, debug)`
Processamento paralelizado de robôs:
- Cada robô busca em sua ROI predita
- Aplicação de fallback coordenado

### 2. `Ball` - Classe da Bola (Atualizada)

**Modelo de Estado Kalman**:
```python
Estado: [x, y, θ, vx, vy] (5 dimensões)
Medição: [x, y, θ]
```

**Características**:
- θ derivado exclusivamente do movimento (não medido)
- ROI adaptativa baseada na covariância do Kalman
- Suporte a predição sem alterar estado do filtro

### 3. `Robot` - Classe dos Robôs (Atualizada)

**Novo Modelo EKF**:
```python
Estado: [x, y, θ, v_esquerda, v_direita, ω] (6 dimensões)
Medição: [x, y, θ]
```

**Características**:
- **Modelo de movimento diferencial** não-linear
- Jacobiano computado para EKF
- Velocidades das rodas no estado do filtro
- ROI baseada em covariância predita

### 4. `TreeColors` - Nova Classe de Gerenciamento de Cores

**Funcionalidades**:
```python
# Cadastro hierárquico de cores
tree.add_robot(team_id, robot_id, main_hsv, primary_hsv, secondary_hsv)

# Busca inteligente por cores
match = tree.find_by_colors(main_candidate, primary_candidate, secondary_candidate)
```

**Vantagens**:
- Tolerância a faixas de Hue circulares
- Sistema de scoring por similaridade
- Fallback para combinações parciais
- Debug visual integrado

## 🔧 Sistema de Coordenadas e Transformações

### Pipeline de Transformação (Atualizado)

```python
# Exemplo com fallback de predição
if detection_success:
    xv, yv = self.transformPoint(np.array([xb, yb]))        # px_virtual  
    xcm, ycm = self.getPointVirtual(np.array([xv, yv]))     # cm
    self.ball.setPosition(xcm, ycm, rb, timestamp)          # atualiza KF
else:
    # Usa predição do Kalman sem atualizar filtro
    self._predict_ball_fallback(timestamp, mark_detected=False)
```

## 🎨 Sistema de Cores e Detecção

### Nova Arquitetura de Cores

#### `TreeColors` - Sistema Hierárquico
```python
# Estrutura de armazenamento
_store: {
    (team_id, robot_id): {
        'colors': {
            'main': [H, S, V],
            'primary': [H, S, V], 
            'secondary': [H, S, V]
        },
        'bounds': { ... },  # Limites pré-computados
        'tolerances': { ... }
    }
}
```

#### Pipeline de Identificação
1. **Filtro por Bounds**: Verificação rápida por limites HSV
2. **Scoring por Similaridade**: Métrica combinada (Hue, Saturação, Valor)
3. **Fallback Parcial**: Match com main + primary se combinação completa falhar

### Detecção de Robôs Aprimorada

```python
# Novo fluxo com TreeColors
match = self.colorTree.find_by_colors(main_color, primary_color, secondary_color)
if match and match['robot_id'] == expected_id:
    bot.updatePosition(xcm, ycm, direction, window, timestamp)
```

## 🔮 Filtro de Kalman Avançado

### 🤖 Modelo EKF para Robôs

**Equações de Movimento Diferencial**:
```python
def _non_linear_motion_model(self, state, dt):
    x, y, theta, vL, vR, omega = state[:, 0]
    v = (vL + vR) / 2.0
    
    x_new = x + v * dt * np.cos(theta)
    y_new = y + v * dt * np.sin(theta) 
    theta_new = theta + omega * dt
    # ... manter velocidades constantes
```

**Jacobiano**:
```python
def _jacobian_motion_model(self, state, dt):
    # Matriz 6x6 com derivadas parciais
    # Inclui termos de acoplamento entre posição e orientação
```

### ⚽ Kalman para Bola

**Modelo Linear Simplificado**:
```python
F = np.array([
    [1, 0, 0, dt, 0],   # x
    [0, 1, 0, 0, dt],   # y  
    [0, 0, 1, 0, 0],    # theta
    [0, 0, 0, 1, 0],    # vx
    [0, 0, 0, 0, 1]     # vy
])
```

## 🔍 Sistema de Detecção Filtrada

### Fluxo Otimizado de Rastreamento

1. **Predição do ROI**:
   ```python
   roi = self.predictRobot(img_shape, team, robot_id, timestamp)
   # Baseado na covariância do Kalman (scale_std = 3)
   ```

2. **Busca em Janela Reduzida**:
   ```python
   window = img[y0:y0+h0, x0:x0+w0]
   found = self.search_bot(img, roi, team, bot_id, timestamp, debug)
   ```

3. **Fallback Inteligente**:
   - Predição do Kalman sem atualização do filtro
   - Manutenção da direção e velocidade atuais
   - Status `detected = False` para indicar predição

### Benefícios da Nova Abordagem

- **Performance**: Redução de ~80% na área de busca
- **Robustez**: Continua operando durante oclusões parciais
- **Precisão**: ROI adaptativa baseada na incerteza do Kalman
- **Consistência**: Transição suave entre detecção e predição

## ⚡ Otimizações e Performance

### 1. Gerenciamento de Estado Aprimorado
```python
def resetExecutionState(self):
    """Reset seletivo - preserva Kalman e histórico"""
    # Limpa apenas estado temporário do frame
    # Mantém: kalman_state, position_history, velocity
```

### 2. Controle de Falhas em Campo
```python
def _check_field_reset(self):
    """Reset completo após múltiplas falhas consecutivas"""
    if self.fieldDetectionFailCount >= self.maxFieldFailures:
        self._resetVs()  # Recria todos os objetos
```

### 3. Processamento Paralelo
```python
# Busca paralela de robôs usando ThreadPool
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(self._process_single_bot, ...) for bot in team]
```

## 🎯 Configuração

### Novos Parâmetros do Sistema
```python
# TreeColors
hue_tolerance = 10      # Tolerância circular do Hue
score_threshold = 200.0 # Limiar de similaridade

# Kalman ROI
scale_std = 3          # Multiplicador do desvio padrão para ROI
min_roi_size = 12      # Tamanho mínimo da janela de busca
```

## 📊 Saída do Sistema

### Estrutura de Dados Atualizada
```python
objects = {
    ID_Objects.ALLIES: [
        robotAllyG,  # .detected = True/False (deteção real/predição)
        robotAlly1, 
        robotAlly2
    ],
    ID_Objects.ENEMIES: [...],
    ID_Objects.BALL: ball_object,
    'timestamp': self.dT,
    'processing_mode': 'full|filtered'  # Novo campo
}
```

### Novas Imagens de Debug
- `binaryBall`: Bola com blob da detecção atual
- `virtualImg`: Setas de direção e orientação dos robôs
- **TreeColors Debug**: Impressão hierárquica das cores cadastradas

## 🚀 Uso Básico (Atualizado)

```python
# Inicialização com nova árvore de cores
config = EConfig()
vs = VisionSystem(config=config, debug=True)

# Processamento inteligente (auto-ajustável)
while True:
    img = capture.getImage()
    result = vs.processImg(img, debug=True)
    
    # Obter dados com informação de modo
    objects = vs.getObjects()
    ball = objects[ID_Objects.BALL]
    
    if ball.detected:
        print(f"Bola detectada: {ball.position}")
    else:
        print(f"Bola predita: {ball.position}")
```

## ⚠️ Considerações Técnicas Atualizadas

### Kalman e Modelos de Movimento
1. **Robôs**: EKF com modelo diferencial - requer calibração de parâmetros (axle_length)
2. **Bola**: Modelo linear - θ derivado do movimento (pode acumular erro)
3. **Inicialização**: Primeiros frames críticos para convergência do filtro

### TreeColors e Identificação
1. **Calibração**: Requer configuração precisa das cores principais e secundárias
2. **Iluminação**: Tolerâncias devem ser ajustadas para condições de luz
3. **Oclusão**: Sistema funciona com cores parciais (main + primary)

### Performance em Tempo Real
1. **Fase Inicial**: 60 frames de processamento completo para estabilizar Kalman
2. **Operação Normal**: 80-90% de redução no processamento com detecção filtrada
3. **Recuperação**: Reset automático após falhas persistentes

### Validação Recomendada
- Testar transição entre detecção e predição
- Validar ROIs em diferentes velocidades
- Verificar matching de cores sob variação de iluminação
- Monitorar covariância do Kalman para detectar divergência

Este sistema fornece uma base robusta e eficiente para detecção em tempo real, com arquitetura adaptativa que balanceia precisão e performance em ambientes dinâmicos de competição robótica.