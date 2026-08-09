# VSSS System ⚽️🤖

Bem-vindo ao VSSS System — um kit modular de visão computacional e controle para Very Small Size Soccer (VSSS).  
É prático, extensível e feito para acelerar experimentos com visão, controle e comunicação de robôs pequenos.

**Por que o VSSS System?**

* Pipeline de visão pronto para detectar campo, bola e robôs em tempo real.
* Emulador multi-thread para orquestrar captura, visão, controle e comunicação.
* Interface gráfica (Tkinter) para experimentar parâmetros sem mexer em código.
* Suporte opcional a GPU (CUDA) para quem quer rodar mais rápido.
* Rastreamento robusto com Filtro de Kalman (EKF para robôs, KF linear para bola).
* Detecção adaptativa com ROI e fallback inteligente.

---

## 🏗️ Arquitetura e Módulos

O projeto é organizado em módulos para facilitar manutenção, testes e extensibilidade:

1. **Detector (`src/modules/VisionSys/detectorV2.py`)** – **v3.3.22 (BETA)**

   * Núcleo da visão computacional.
   * Processa imagens para detectar campo, bola e robôs aliados/inimigos.
   * Integração com filtro de Kalman (EKF para robôs, KF linear para bola).
   * Utiliza `TreeColors` para identificação de robôs por cores HSV.
   * **Modularização completa** da detecção de jogadores em 3 partes reutilizáveis:
     - `DetectPlayerCandidates` – extrai candidatos crus (blobs) da imagem.
     - `ResolveCollisionBlob` – trata blobs com múltiplos robôs via erosão iterativa + predição do Kalman.
     - `AssociatePlayerCandidates` – identifica a qual robô específico cada candidato pertence e atualiza o filtro.
   * **Gating de distância** (`MAX_JUMP_CM`) para rejeitar medições implausíveis.
   * **Watchdog** com reset automático do filtro após perda prolongada (10 frames para bola, 15 para robôs).
   * **Pipeline híbrido**: detecção completa nos primeiros frames, depois rastreamento filtrado (ROI adaptativa).

2. **Emulador (`src/modules/emulator`)**

   * Coordena threads de captura, processamento e comunicação.
   * Mantém filas de frames e eventos para garantir execução estável.
   * Exibe métricas de performance e FPS em tempo real.

3. **Interface Gráfica / UI (`src/ui`)**

   * Cards de informação, ajustes de parâmetros e visualização de ROI.
   * Menu de configuração de cores, thresholds, offsets e modos de captura.
   * Mostra imagens de debug, máscaras binárias e posições detectadas.

4. **Assets e Configurações (`src/data`, `src/images`)**

   * Imagens de referência, logos, planos de fundo e presets.
   * Arquivos de configuração para cores, tamanhos e parâmetros de visão.

5. **Entrada Principal (`main.py`)**

   * Inicializa módulos, threads e UI.
   * Seleciona modo de execução: câmera, vídeo ou imagem estática.
   * Carrega configurações e inicia loop principal do sistema.

Essa divisão modular permite que cada componente seja testado ou substituído independentemente, por exemplo:

* Trocar o módulo de captura de vídeo sem alterar a detecção.
* Atualizar o detector para novas cores de robôs sem modificar o emulador.
* Acrescentar estratégias ou comunicação adicional como módulos separados.

---

## 🧠 Filtro de Kalman

O **Filtro de Kalman** é utilizado no VSSS System para **predizer posições futuras da bola e dos robôs**, reduzindo o ruído das medições de visão e permitindo criar **ROIs mais precisas**.

**Características principais:**

* **Robôs (EKF)**: modelo de estado com 6 dimensões `[x, y, θ, vL, vR, ω]` – captura melhor a dinâmica diferencial.
* **Bola (KF linear)**: modelo de estado com 5 dimensões `[x, y, θ, vx, vy]` – posição e velocidade linear.
* **ROI dinâmica**: a posição prevista e a covariância definem uma janela de busca ajustável (`scale_std=3`).
* **Gating de distância**: rejeita medições cujo salto exceda `MAX_JUMP_CM` (padrão 50 cm), evitando correções bruscas.
* **Watchdog**: contador de frames perdidos; ao atingir o limite (10 para bola, 15 para robôs), o filtro é reiniciado na próxima detecção.
* **Fallback**: durante perda de detecção, a predição do Kalman mantém a estimativa sem atualizar o filtro, garantindo continuidade.

**Benefícios:**

* Processamento mais rápido, pois apenas pequenas regiões (ROIs) são analisadas.
* Menor chance de "sumir" um objeto em frames consecutivos.
* Estimativas suaves e robustas mesmo com detecções ruidosas.
* Integração direta com `search_bot`, `search_ball` e `filtered_detection`.

---

## 🔍 Detecção com `filtered_detection`

O método `filtered_detection` é o **pipeline central de rastreamento** do VSSS System, acionado após o aquecimento do Kalman. Ele combina:

1. **Entrada da imagem recortada** (`fieldReduce`) com base no ROI do campo.
2. **Predição do Kalman** para cada objeto (bola e robôs) → ROI em pixels via `GetRoiImg`.
3. **Busca localizada**:
   - `SearchBall` – detecta a bola dentro da ROI prevista.
   - `SearchBots` – para cada robô, chama `DetectBotInRoi` (que usa o protocolo geométrico de tags T1/T2 para localização precisa).
4. **Atualização dos filtros**:
   - Se encontrado: `updatePosition` (correção do Kalman).
   - Se não encontrado: incrementa contador de falhas; se exceder limite, marca para reset.
5. **Fallback**: se perdido, usa a predição do Kalman para manter a estimativa visual.
6. **Atualização de máscaras globais** (`binaryBall`, `binaryPlayers`, `binaryAllies`) para debug.
7. **Desenho de ROIs e objetos** na imagem final (quando `debug=True`).

**Fluxo resumido:**

* Inicializa contadores e máscaras zeradas.
* Recorta o campo a partir do ROI salvo (`viewCapture.cooVetor`).
* Executa `PredictBall` → `SearchBall` → atualiza Kalman da bola.
* Para cada robô (aliados e inimigos):
  - Executa `PredictRobot` → `SearchBots` (que chama `DetectBotInRoi`).
  - Se encontrado, atualiza Kalman com `updatePosition`.
  - Se não, chama `HandleRobotLoss` (predição ou reset).
* Desenha todos os objetos detectados e as ROIs (se debug).
* Retorna a imagem processada.

**Benefícios:**

* Pipeline **leve e seguro**, protegido contra travamentos (uso de `SafeCall`).
* Processamento **focado apenas nas regiões de interesse**, aumentando FPS.
* Compatível com múltiplos modos de execução: câmera ao vivo, vídeo ou imagens estáticas.
* Tratamento de colisões integrado (blobs com múltiplos robôs) via `ResolveCollisionBlob`.

---

## ⚙️ Detalhes da Modularização da Detecção de Jogadores (v3.3.22)

A função `DetectPlayers` foi dividida em três partes reutilizáveis, facilitando manutenção e permitindo reaproveitamento em janelas de qualquer tamanho (ex.: ROIs):

### Parte 1 – `DetectPlayerCandidates`
- Gera máscara genérica de objetos (excluindo a bola).
- Extrai contornos e classifica por área estimada.
- Se o blob corresponde a um único robô, constrói candidato via `_BuildPlayerCandidate`.
- Se o blob é grande demais (colisão), delega para `ResolveCollisionBlob`.

### Parte 2 – `ResolveCollisionBlob`
- Separa o blob fundido em múltiplos centros usando **erosão iterativa** (`_SplitBlobByErosion`).
- Se ainda faltarem centros, complementa com a **posição prevista pelo Kalman** dos robôs mais próximos (`_GetKalmanPredictedCenters`).
- Retorna até `n_est` centros (x, y) em coordenadas locais.

### Parte 3 – `AssociatePlayerCandidates`
- Para inimigos: pareamento por posição (última conhecida) com gating de distância.
- Para aliados: usa `DetectAllyRobot` (verificação das cores primária/secundária) + gating.
- Atualiza o filtro de Kalman com `setPosition` (primeira medição) ou `updatePosition` (demais).
- Gerencia a árvore de cores (`TreeColors`) para aprendizado contínuo das cores dos inimigos.

Essa separação permite:
- Reutilizar `DetectPlayerCandidates` em ROIs menores (ex.: busca dentro da ROI prevista pelo Kalman).
- Evoluir o tratamento de colisões sem afetar a identificação.
- Estender a lógica de identificação (ex.: usar `IdentifyCandidateByColor` como critério extra) sem tocar no pipeline de blob.

---

## 🚀 Quick Start (rápido)

1. Clone o repositório:
   ```bash
   git clone <seu-repositorio>