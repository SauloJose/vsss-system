# VSSS System 3.4.21

Sistema modular de visao computacional e controle para futebol de robos Very
Small Size Soccer (VSSS). O projeto recebe imagens de camera, video ou arquivo,
localiza campo, bola e robos, e disponibiliza dados para a interface e os
modulos de comunicacao.

## Funcionalidades

- Deteccao de campo, bola e ate seis robos usando OpenCV.
- Identificacao de robos por tags coloridas em HSV e pela estrutura `TreeColors`.
- Rastreamento com filtro de Kalman e janelas de busca (ROI) adaptativas.
- Tratamento de candidatos agrupados, incluindo separacao de blobs por erosao.
- Interface Tkinter, emulador e modulos de comunicacao.
- Execucao em CPU; recursos CUDA/OpenCL dependem da compilacao do OpenCV e do
  hardware disponivel.

## Organizacao do projeto

```text
.
|-- main.py                         # Inicializacao da aplicacao desktop
|-- setup.py                        # Metadados para instalacao com pip
|-- run_pytest.py                   # Atalho para executar toda a suite de testes
|-- src/
|   |-- app.py                      # Composicao da interface e do emulador
|   |-- imports.py                  # Importacoes compartilhadas pela aplicacao
|   |-- data/                       # Configuracoes, imagens e dados de execucao
|   |-- modules/
|   |   |-- VisionSys/
|   |   |   |-- detectorV2.py       # Pipeline de deteccao e rastreamento
|   |   |   `-- components/         # Bola, robos, campo, geometria e cores
|   |   |-- communication/          # Comunicacao com dispositivos e simuladores
|   |   |-- control/                # Comandos, estados e protocolos
|   |   `-- emulator/               # Orquestracao dos componentes
|   `-- ui/                         # Janelas, menus e componentes visuais
`-- tests/                          # Testes unitarios e de integracao
```

### Componentes de visao

- `detectorV2.py`: classe `VisionSystem`, deteccao completa e rastreamento por
  ROI, desenho de debug e associacao dos candidatos aos objetos.
- `components/ball.py`: estado, geometria, velocidade e filtro da bola.
- `components/robot.py`: estado, orientacao, geometria e filtro dos robos.
- `components/field.py` e `components/viewcapture.py`: campo e captura/visao da
  imagem.
- `components/objects.py`: identificadores, primitivas geometricas e utilitarios
  compartilhados.
- `components/combination.py`: cadastro e correspondencia das cores HSV por
  equipe e robo (`TreeColors`).

## Filtros de Kalman

Os filtros sao mantidos pelos proprios objetos e usam timestamps em segundos.
As medidas recebidas pelo detector atualizam o estado; as predicoes sao usadas
para posicionar as ROIs e manter a estimativa visual quando uma deteccao falha.

### Bola: KF linear

O estado tem quatro dimensoes, `[x, y, vx, vy]`. O modelo de velocidade
constante propaga posicao e covariancia entre quadros; a correcao usa as
medidas de posicao `[x, y]`. A orientacao e derivada da velocidade filtrada.
`Ball.predict(timestamp)` calcula uma predicao sem alterar o estado e
`Ball.get_roi(...)` devolve uma janela em centimetros dimensionada pela
incerteza do filtro.

### Robos: EKF nao linear

O estado tem cinco dimensoes, `[x, y, theta, v, omega]`, onde `v` e a
velocidade linear e `omega` a velocidade angular. O modelo de movimento
diferencial/unicycle e linearizado pela jacobiana; a diferenca angular da
medida e normalizada para evitar descontinuidades em `-pi`/`pi`. O filtro
mantem a covariancia usando a forma de Joseph na correcao.
`Robot.predict(...)` e `Robot.predict_with_cov(...)` permitem consultar o
estado futuro, e `Robot.get_roi(...)` retorna a janela de busca em centimetros.

O detector converte essas janelas para pixels, busca primeiro na ROI prevista
e pode tentar uma ROI de recuperacao centrada na ultima posicao medida. O
gating rejeita saltos acima de `MAX_JUMP_CM` (80 cm). Contadores de quadros
perdidos controlam o reset: os limites atuais sao 360 quadros para bola e
robos; a deteccao completa e periodicamente reexecutada para corrigir deriva.
Esses parametros pertencem a `VisionSystem` e podem evoluir com a calibracao.

## Deteccao e associacao de objetos

O modo de imagem executa o pipeline completo. Em fluxo continuo, a deteccao
alimenta os filtros e o modo rastreado procura a bola e cada robo nas ROIs
previstas. A associacao de jogadores esta dividida em etapas reutilizaveis:

1. `DetectPlayerCandidates` extrai candidatos e seus dados geometricos.
2. `ResolveCollisionBlob` tenta separar objetos sobrepostos por erosao e usa
   as posicoes estimadas para auxiliar a separacao.
3. `AssociatePlayerCandidates` associa candidatos a robos aliados/inimigos
   usando cor e posicao, aplicando o gating antes de atualizar os objetos.

Quando um objeto nao e encontrado, o detector conserva a predicao e incrementa
seu contador de falhas. Ao ultrapassar o limite de perda, o filtro e marcado
para reinicializacao na proxima medida valida. As rotinas de debug podem
desenhar mascaras, ROIs e objetos detectados.

## Instalacao

Requer Python 3.10 ou superior. A instalacao local instala os modulos e suas
dependencias de execucao:

```powershell
python -m pip install .
```

Para instalar tambem as dependencias de teste:

```powershell
python -m pip install ".[test]"
```

Para desenvolver no repositorio, use instalacao editavel:

```powershell
python -m pip install -e ".[test]"
```

A interface desktop usa Tkinter, que deve estar disponivel na distribuicao do
Python. Para iniciar a aplicacao completa, execute `python main.py` a partir
da raiz do repositorio; configuracoes e imagens de execucao estao em `src/data`
e sao referenciadas relativamente a essa pasta de trabalho. A aceleracao CUDA
nao e garantida pela instalacao padrao do OpenCV via pip.

## Testes

Execute todos os testes a partir da raiz:

```powershell
python -m pytest -q
```

Os testes dos filtros sao deterministas e nao precisam de camera nem videos:

```powershell
python -m pytest -q tests/test_kalman_models.py
```

O atalho `python run_pytest.py` executa a suite completa usando o mesmo
interpretador Python que iniciou o script. Testes que dependem de arquivos de
video sao ignorados quando esses arquivos nao estao presentes.

## Documentacao adicional

- [Sistema de visao](src/modules/VisionSys/readme.md)
- [Componentes e cores](src/modules/VisionSys/components/readme.md)
- Documentacao de comunicacao e controle nos respectivos diretorios de
  `src/modules/`.
