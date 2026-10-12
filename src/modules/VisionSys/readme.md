# Sistema de Visao

Modulo de deteccao e rastreamento utilizado pelo VSSS System 3.4.21. A
implementacao principal esta em `detectorV2.py`, na classe `VisionSystem`.

## Responsabilidades

- Detectar o campo e transformar coordenadas entre imagem e campo virtual.
- Detectar a bola e seis robos, identificar suas cores e atualizar seus estados.
- Desenhar resultados e informacoes de depuracao.
- Alternar entre deteccao completa e rastreamento localizado em ROIs.

## Fluxo de processamento

`ProcessImg` seleciona o processamento de acordo com o modo de execucao. Uma
imagem isolada passa pela deteccao completa (`Proc`). Em video, as deteccoes
completas alimentam os objetos e os filtros; o rastreamento filtrado consulta
`Ball.get_roi` e `Robot.get_roi`, converte as janelas de centimetros para
pixels e procura os objetos nessas regioes. O detector tambem reexecuta a
deteccao mais ampla periodicamente para corrigir deriva.

Quando uma deteccao nao esta disponivel, a predicao e usada como estimativa
visual sem aplicar uma medida falsa. Os contadores de perda marcam o filtro
para reinicializacao apos 360 quadros consecutivos sem deteccao (bola ou robo).
Saltos maiores que `MAX_JUMP_CM` (80 cm) sao rejeitados na associacao.

## Modelos de movimento

### Bola

`components/ball.py` implementa um KF linear de velocidade constante com estado
`[x, y, vx, vy]`. A observacao e a posicao `[x, y]`; direcao e orientacao sao
derivadas da velocidade filtrada. `predict(timestamp)` nao altera o estado
armazenado, e `get_roi(...)` retorna uma janela em centimetros baseada na
incerteza projetada.

### Robos

`components/robot.py` implementa um EKF nao linear com estado
`[x, y, theta, v, omega]`. O modelo de movimento propaga a posicao e orientacao
com velocidades linear e angular. As diferencas de angulo sao normalizadas e a
covariancia e corrigida usando a forma de Joseph. `predict(...)`,
`predict_with_cov(...)` e `get_roi(...)` permitem obter predicoes e a janela
adaptativa sem exigir que o detector conheca a implementacao interna do filtro.

## Deteccao de jogadores

O detector separa a identificacao em etapas:

- `DetectPlayerCandidates`: extrai candidatos a partir da imagem/ROI.
- `ResolveCollisionBlob`: trata blobs de multiplos robos com erosao iterativa e
  estimativas de posicao.
- `AssociatePlayerCandidates`: associa candidatos a equipe e identificador por
  cor e posicao, valida saltos e atualiza os filtros.

`SearchBots`, `DetectBotInRoi` e `SearchBall` sao usados na busca localizada.
Quando a ROI prevista nao recupera um robo, `PredictRobotRecovery` permite uma
tentativa baseada na ultima posicao medida.

## Componentes relacionados

Veja [components/readme.md](components/readme.md) para os objetos de dominio,
geometria e cadastro de cores. A suite esta em `tests/`; os testes dos modelos
de Kalman podem ser executados com:

```powershell
python -m pytest -q tests/test_kalman_models.py
```
