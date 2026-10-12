# Componentes da Visao

Os componentes separam estado dos objetos, geometria e identificacao por cor:

| Arquivo | Responsabilidade |
| --- | --- |
| `ball.py` | Estado e geometria da bola, velocidade e KF linear de quatro estados. |
| `robot.py` | Estado dos robos, orientacao, cores e EKF nao linear de cinco estados. |
| `field.py` | Estado geometrico e atualizacao do campo detectado. |
| `viewcapture.py` | Captura e representacao da area de imagem observada. |
| `objects.py` | Identificadores, primitivas geometricas e utilitarios compartilhados. |
| `combination.py` | Cadastro e correspondencia de cores HSV por equipe e robo (`TreeColors`). |

## Objetos rastreados

`Ball` armazena posicao, raio, velocidade e estado de deteccao. Seu filtro
prediz `[x, y, vx, vy]` a partir de medidas de posicao. `Robot` armazena
posicao, direcao, dimensoes, cores e status; seu EKF usa o estado
`[x, y, theta, v, omega]`. Ambos oferecem metodos de predicao e ROI para que o
detector busque os objetos sem duplicar a logica de movimento.

O modulo `objects.py` tambem define identificadores (`ID_Robots`, `ID_Team`,
`ID_Objects`), tipos de geometria e objetos geometricos como `Point2D`, `Quad`,
`Circle`, `BorderBox` e `AreaField`. Ele contem utilitarios usados pela
interface e pela captura, portanto nao representa apenas o modelo de bola e
robos.

## Cadastro de cores

`TreeColors` fica em `combination.py`. Ele armazena uma combinacao de cores HSV
associada a `(team_id, robot_id)`, permite atualizar e consultar o cadastro e
procura correspondencias usando bounds e uma pontuacao de distancia de cor.
Os limites do Hue consideram sua natureza circular no espaco HSV do OpenCV.

```python
from modules.VisionSys.components.combination import TreeColors

colors = TreeColors()
colors.add_robot(
    team_id=0,
    robot_id=0,
    main_hsv=[60, 200, 200],
    primary_hsv=[50, 180, 180],
    secondary_hsv=[40, 150, 150],
)
match = colors.find_by_colors(
    [62, 198, 210],
    [48, 175, 190],
    [41, 152, 148],
)
```

## Unidades

- Coordenadas do filtro e das ROIs: centimetros no campo virtual.
- Angulos: radianos.
- Timestamps passados aos filtros: segundos.
- Coordenadas de desenho/captura: pixels, convertidos pelo `VisionSystem`.
