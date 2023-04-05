---
title: "MEFE - Ejercicio 11, guía 1"
date: 2023-04-03T18:40:14-03:00
draft: true
summary: "¿Cuál es la probabilidad de que no te toque el ancho de espadas en un partido de truco de 4 personas y 15 manos? ¿Y de que no salga en todo el partido?"
author: ""
math: true
image: "truco-matematicos.png"
---

# 11)

#### Considere un partido de truco entre cuatro jugadores que dura 15 manos. Encuentre la probabilidad de que:



#### a) a un dado jugador nunca le toque el ancho de espadas [Rta: 0.31]



#### b) el as de espadas no salga en todo el partido [Rta: 0.0047]


#### Solución:



Primero lo hago para una mano. Una mano de un partido de truco de 4 personas sería un conjunto de 15 cartas donde no importa el orden. Puedo pensar a las cartas como números del 1 al 40, en ese caso, el conjunto de todas las posibles manos es

$$\Omega = \\{ (a_1, a_2, ... , a_{12}) \mid a_i \in [1, 40], a_i \neq a_j \forall i, j \\}$$

entonces la cantidad de posibles manos es

$$|\Omega| = 40 \times 39 \times ... \times 29 = \frac{40!}{28!}.$$



Ahora, en caso de que salga el ancho **en toda la mano**, una de las cartas ya está definida. Si pienso que el ancho es la carta 1, entonces el conjunto de casos en los que sale el ancho es

$$A = \\{ (1, a_2, a_3, \ldots , a_{12}), (a_1, 1, a_3, \ldots , a_{12}), \ldots , (a_1, \ldots , a_{11}, 1) \mid a_i \in [1, 40], a_i \neq a_j \forall i, j\\},$$

así que el cardinal de A es

$$|A| = 12 \times \frac{39!}{28!},$$

por lo que la probabilidad de que salga el ancho en una mano es

$$P(A) = \frac{|A|}{|\Omega|} = \frac{12}{40} = 0.3,$$

que es lo esperado intuitivamente.



Si lo simulo:

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python
def jugar_manos(n_manos, n_cartas, ancho=1):
    manos = np.repeat(0, repeats=n_cartas*n_manos).reshape(n_manos, n_cartas)
    for i in range(manos.shape[0]):
        manos[i,:] = np.random.default_rng().choice(np.arange(1, 41), size=12, replace=False)
    freq_ancho = np.sum(np.count_nonzero(manos == 1, axis=1))
    return manos, freq_ancho/manos.shape[0]

n_manos = 15000
n_cartas = 12
manos, p_a = jugar_manos(n_manos, n_cartas)
print(p_a)
```

Ahora hay que ver la probabilidad de sacar el ancho para una dada mano. La cantidad de posibles manos de 3 cartas esta dada por el cardinal del conjunto

$$\Omega_B = \\{(b_1, b_2, b_3) \mid b_i \in \\{a_i\\}_{i=1}^{i=12}, b_i \neq b_j \forall i, j \\},$$

que es 

$$|\Omega_B| = 12 \times 11 \times 10.$$



Si pedimos que aparezca el ancho (para esto se tiene que haber dado A) obtenemos el conjunto

$$B = \\{ (1, b_2, b_3), (b_1, 1, b_3), (b_1, b_2, 1) \mid b_i \in \\{a_i\\}_{i=1}^{i=11}, b_i \neq b_j \forall i, j \\},$$

que tiene cardinal 

$$|B| = 3 \times 11 \times 10,$$

por lo que la probabilidad de sacar el ancho dado que el ancho salió en la mano es

$$P(B|A) = \frac{|B|}{|\Omega_B|} = 3/12 = 0.25.$$



Podemos verificarlo numéricamente usando las manos que generamos antes:
```python
manos_con_ancho = manos[np.count_nonzero(manos == 1, axis=1) == 1]
p_ba = np.sum(np.count_nonzero(manos_con_ancho[:, :3] == 1, axis=1))/manos_con_ancho.shape[0]
print(freq_ancho_mano)
```

Entonces, para que te toque el ancho en una mano, primero tenés que tener la suerte de que el ancho salga en la mano (P(A)), y luego, de que te toque a vos dado que salió en la mano (P(B|A)). Entonces, la probabilidad de que te toque el ancho en una mano es

$$P(ancho) = P(B|A)P(A) = 0.25 \times 0.3 = 0.075.$$



Podemos verificarlo nuevamente con los mismos datos:
```python
p_ancho = np.sum(np.count_nonzero(manos[:,:3] == 1, axis=1))/manos.shape[0]
print(p_ancho)
```

Ahora, esa es la probabilidad de que a un dado jugador le toque el ancho de espadas en **una mano**. Por lo tanto, la probabilidad de que **no** le toque es 

$$\tilde{P}(ancho)=1 - P(ancho) = 0.925.$$

Entonces, la probabilidad de que **no** le toque en 15 manos, será

$$\tilde{P}(ancho)^{15} \approx 0.31$$
**b)**

La probabilidad de que el ancho no salga en todo el partido es la probabilidad de que no salga en una mano, elevado a la 15:

$$\tilde{P}(A)^{15} = (1 - P(A))^{15} = 4.7 \times 10^{-4}$$
