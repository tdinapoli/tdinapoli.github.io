---
title: "MEFE - Ejercicio 9, Guía 2"
date: 2023-04-17T09:25:31-03:00
draft: false
summary: ""
author: ""
math: true
image: "portada.png"
---

## Enunciado
#### 9)

Una fabrica produce integrados, de los cuales el 20% son defectuosos y los comercializa en cajas de 10. Un comprador quiere rechazar las cajas que contengan más de 2 chips defectuosos, es decir, más que la especificación del fabricante. Para ganar tiempo, en vez de probar todos los chips decide implementar el siguiente test. De cada caja toma 6 chips al azar: (i) si ninguno es malo, acepta la caja; (ii) si uno solo es malo, revisa el resto de la caja; (iii) si 2 o más son malos, devuelve la caja al fabricante.

**a)** ¿Qué fracción de las cajas tendrán más de 2 chips defectuosos? [Rta: 0.322]  
**b)** ¿En qué fracción de las cajas deberá probar los 10 integrados? [Rta: 0.3932]  
**c)** ¿Cuál es la probabilidad de que haya 3 chips malos en una caja aceptada? [Rta: 0.0114]

<hr>

## Solución
<br></br>

### a) ¿Qué fracción de las cajas tendrán más de 2 chips defectuosos? [Rta: 0.322]
Para armar las cajas, el fabricante toma uno de sus integrados y lo mete adentro. Luego otro, y lo vuelve a meter en la caja. Si cada caja tiene $N$ integrados, repite esta acción $N$ veces. Con un 20% de chances, cada integrado que agarre el fabricante al armar la caja será defectuoso ($\overline{B}$), y por lo tanto con un 80% de chances, será bueno ($B$).



Entonces la probabilidad de tomar, por ejemplo, 10 integrados malos, es $p^{10}=0.2^{10}$, o 10 buenos sería $(1-p)^{10} = 0.8^{10}$. Pero en general, la probabilidad de sacar $n$ malos y $N-n$ buenos es $p^n (1-p)^{N-n}$. Da igual el orden en el que el fabricante ponga los integrados en la caja, al fin y al cabo la cantidad de integrados malos es la misma si agarró los malos primero y los buenos después, o al revés. Por eso, hay que multiplicar por la cantidad de formas que hay de sacar $n$ malos y $N-n$ buenos, que es el combinatorio ${N \choose n}$. 

En conclusión, la probabilidad de que una caja con $N$ integrados tenga $n$ malos dado que la probabilidad de poner uno malo en la caja es $p$ es

$$B(n \mid N, p) = {N \choose n} p^n (1-p)^{N-n},$$

entonces la probabilidad de que tenga 2 o más con $N=10$ y $p=0.2$ es

$$\sum_{n=3}^{n=10} {10 \choose n} 0.2^n 0.8^{10-n} = 0.322.$$



Se puede chequear con código de dos formas:

1. armando las cajas y contando cuántas tienen más de 2 integrados malos (experimental)

1. haciendo lo mismo pero tomando los valores random directamente de una binomial (binomial)
```python
import random
import numpy as np
import math

# Defino la binomial
def bin(N, n, p):
    return math.comb(N, n) * p**n * (1-p)**(N-n)

# Parámetros
p = 0.2
N = 10
tolerancia_minima = 2
n_cajas_compradas = 100000

# Experimental
cajas_compradas = np.array([[(random.random() < p)*1 for i in range(N)] for j in range(n_cajas_compradas)])
malos_por_caja = np.count_nonzero(cajas_compradas, axis=1)
cajas_con_mas = np.count_nonzero(malos_por_caja > tolerancia_minima)
print(f"Propoción de cajas armadas con más de {tolerancia_minima} integrados malos:\n{cajas_con_mas/n_cajas_compradas}\n")

# Binomial
malos_por_caja_binomial = np.random.binomial(N, p, size=n_cajas_compradas)
cajas_con_mas_binomial = np.count_nonzero(malos_por_caja_binomial > tolerancia_minima)
print(f"Propoción de cajas binomial con más de {tolerancia_minima} integrados malos:\n{cajas_con_mas_binomial/n_cajas_compradas}\n")

# Teórico
teorico = np.sum([bin(N, n, p) for n in range(3, 11)])
pm3 = teorico
print(f"El resultado teórico es:\n{teorico}")
```

<br></br>

### b) ¿En qué fracción de las cajas deberá probar los 10 integrados? [Rta: 0.3932]

El comprador debe probar todos los integrados sólo en el caso en el que sea positiva la condición (ii) de su test, es decir, cuando de los 6 que sacó al azar de la caja, sólo 1 es malo. Sabemos que para que eso pase una caja puede tener a lo sumo 5 integrados malos, porque si tuviera 6 o más seguro saca 2 o más malos al sacar 6 al azar, por lo que descartaría la caja de inmediato. En el otro extremo, la caja debe tener al menos 1 malo, pues si no la aceptaría de inmediato. En general, queremos saber cuál es la probabilidad de sacar $k$ integrados malos al extraer $n$ integrados de una caja que contiene $N$ en total, dado que $m$ de ellos son malos. 

Ningún integrado tiene más probabilidad de ser extraído que otro, así que podemos calcular la probabilidad contando casos favorables sobre casos totales. Acá, la cantidad de casos totales es la cantidad de formas de tomar $n$ objetos de un grupo de $N$, que es ${N \choose n}$. Los casos "favorables" serían los casos en los que $k$ de los $n$ que agarramos son malos, y por lo tanto $n-k$ son buenos. La cantidad de formas de extraer $k$ de los $m$ malos que tiene la caja es ${m \choose k}$, y la cantidad de formas de extraer los $n-k$ buenos restantes es ${N - m \choose n - k}$, por lo que la probabilidad de obtener $k$ malos al extraer $n$ integrados de una caja de $N$ con $m$ malos está dada por la hipergeométrica



$$H(k \mid n, m, N) = \frac{{m \choose k}{N-m \choose n-k}}{{N \choose n}}.$$



Hay que recordar que esta probabilidad tiene en cuenta que hay $m$ malos en la caja, pero eso no es dato para nosotros. Lo que queremos calcular es $P(k=1)$, pero la cuenta de arriba nos dice $P(k \mid m)$. Para obtener lo que queremos hace falta multiplicar por la probabilidad de obtener $m$ malos en una caja, es decir $P(k)=P(k \mid m)P(m)$, que por lo que dedujimos en el **a)** es



$$P(k) = {N \choose m} p^m (1-p)^{N-m} \frac{{m \choose k}{N-m \choose n-k}}{{N \choose n}}.$$



En particular, a nosotros nos interesa saber la probabilidad de sacar $k=1$ malo al sacar un subconjunto de $n=6$ de una caja de $N=10$ integrados, dado que puede haber $m=1, 2, 3, 4$ o $5$ integrados malos dentro de ella. Es decir



$$\sum_{m=1}^5 {N \choose m} p^m (1-p)^{N-m} \frac{{m \choose 1}{10-m \choose 6-1}}{{10 \choose 6}} = 0.3932.$$



Ahora para verificar codeamos.



Primero hacemos la cuenta analítica:
```python
# Defino la hipergeométrica:
def hg(k, n, m, N):
    return (math.comb(m, k) * math.comb(N-m, n-k))/math.comb(N, n)

# Constantes
k = 1
n = 6

# Teórico
teorico = np.sum([hg(k, n, m, N)*bin(N, m, p) for m in range(1, 6)])
print(f"El resultado teórico es:\n{teorico}\n")
```

Ahora, como las cajas compradas del punto **a)** ya están hechas de forma aleatoria, puedo reutilizarlas:
```python
subconjuntos_test = cajas_compradas[:, :6] # Tomo el subconjunto de integrados en cada caso
malos_por_subconjunto = np.count_nonzero(subconjuntos_test, axis=1) # Cuento la cantidad de malos por subconjunto
subconjuntos_con_k = np.count_nonzero(malos_por_subconjunto == k) 
print(f"Proporcion de cajas en las que hay que testear todos:\n{subconjuntos_con_k/n_cajas_compradas}")
```

<br></br>

### c) ¿Cuál es la probabilidad de que haya 3 chips malos en una caja aceptada? [Rta: 0.0114]

Nos piden la probabilidad de que haya 3 integrados malos en una caja aceptada. Esto es, dado que la caja **ya fue aceptada**, la probabilidad de que haya 3 integrados malos en ella. Podemos decir que las cajas aceptadas pasaron el test ($+$) y que las cajas rechazadas no ($-$). Así, quedaría que la probabilidad de que haya 3 integrados malos en una caja aceptada es $P(3 \mid +)$, que es lo que nos estan preguntando. No sabemos calcular eso directamente, para calcularlo necesitamos usar el Teorema de Bayes:



$$P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)},$$



donde acá $A=3$ y $B=+$,



$$P(3 \mid +) = \frac{P(+ \mid 3)P(3)}{P(+)}.$$



Acá, $P(3)$ se refiere a la probabilidad de que hayan salido 3 malos en la caja de 10 integrados, y $P(+ \mid 3)$ se refiere al a probabilidad de que la caja sea aceptada luego de aplicar el test dado que tenía 3 integrados malos en ella. 



Sólo nos quedaría ver quién es $P(+)$. Es la probabilidad de que la caja sea aceptada. No sabemos calcularla directamente, pero sabemos calcular $P(+ \mid k)$ y $P(k)$ que la suma sobre $k$ de su producto es 



$$P(+) = \sum_{k=0}^{k=10}P(+ \mid k)P(k).$$



Los valores de $P(k)$ son fáciles de calcular porque son independientes al test. Son la probabilidad de que salgan $k$ integrados en una caja, es decir $P(k) = B(k \mid 10, 0.2)$. Para los otros hay que pensar un poco más. 



Como dijimos antes al resolver el punto **b)**, para $k \geq 5$ todas las cajas serán rechazadas, pues si tienen 5 seguro uno de ellos entrará dentro de los 6 que se remueven para hacer el test, que daría el resultado (ii) y la caja sería rechazada una vez que se analicen los restantes 4 integrados. Si tienen más de 5, seguro dos de ellos entran en los 6 para el test, por lo que la caja sería rechazada inmediatamente. Todo esto se resume en que 



$$P(+ \mid k) = 0 \\; \forall \\; k \geq 5,$$



es decir si hay más de 5 malos en la caja, la probabilidad de que sea aceptada es nula.



Otros dos casos particulares son $k=0$ y $k=1$, ya que si no hay ningún integrado malo entonces el test da (i) y es aceptado. Si hay uno, el test da (i) y es aceptado, o da (ii) y es aceptado luego de chequear que no hay ningún otro integrado en la caja. Por lo tanto, en todos los casos la probabilidad de que una caja sea aceptada dado que tiene 0 o 1 integrados malos es $P(+ \mid 0) = P(+ \mid 1) = 1$.



Ahora nos queda calcular $P(+ \mid 2)$, $P(+ \mid 3)$ y $P(+ \mid 4)$, que no son tan triviales como los otros.



Para $P(+ \mid 2)$, si sale 1(ii) o ninguno(i) malo en el grupo de 6, la caja es aceptada. Sólo en el caso en el que salen 2 en esos 6 la caja es rechazada, por lo tanto podemos calcular la probabilidad de que sea aceptada como 



$$P(+ \mid 2) = 1 - P(- \mid 2) = 1 - H(2 \mid 6, 2, 10) = 0.66...,$$



donde usé la hipergeométrica del punto **b)** para calcular la probabilidad de que salgan 2 malos en el grupo de 6 dado que había 2 malos en la caja de 10 integrados.



Ahora $P(+ \mid 3)$ y $P(+ \mid 4)$ son parecidos. En ambos casos, las cajas son aceptadas sólo si el test da (i). Esto quiere decir que podemos calcular las probabilidades con la hipergeométrica

$$P(+ \mid 3) = H(0 \mid 6, 3, 10) = 0.0047$$
$$P(+ \mid 4) = H(0 \mid 6, 4, 10) = 0.0333...$$

Ya tenemos todo para calcular 

$$\begin{align*} P(+) &= P(0) + P(1) + P(+ \mid 2)P(2) + P(+ \mid 3) P(3) + P(+ \mid 4) P(4) \\\\ &= 0.11 + 0.27 + 0.666... \times 0.3 + 0.0333... \times 0.2 + 0.0047 \times 0.088 \\\\ &= 0.587 \end{align*},$$

y también, para calcular $P(+)$ tuvimos que calcular el numerador de Bayes, $P(+ \mid 3)P(3) = 0.00666...$ por lo tanto el resultado queda

$$P(\text{3 malos en una caja aceptada}) = 0.0114.$$

Podemos calcular esto numéricamente aplicando el test al conjunto de cajas que generamos antes:
```python
# Filtros para los casos del test
filtro1 = np.sum(cajas_compradas[:,:6], axis=1) == 0
filtro2 = np.sum(cajas_compradas[:,:6], axis=1) == 1
filtro_2_aceptados = np.sum(cajas_compradas[filtro2, :], axis=1) <= 2
filtro3 = np.sum(cajas_compradas[:,:6], axis=1) >= 2
filtro_3_malos = np.sum(cajas_compradas, axis=1) == 3

# Número de cajas en cada caso
caso_1 = np.count_nonzero(filtro1)
caso_2 = np.count_nonzero(filtro2)
caso_2_aceptadas = np.count_nonzero(filtro_2_aceptados)
caso_3 = np.count_nonzero(filtro3)
total = caso_1 + caso_2 + caso_3
aceptadas = caso_2_aceptadas + caso_1

# P(+)
pmas = aceptadas/total
# P(3)
p3 = np.count_nonzero(filtro_3_malos)/total
# P(+ | 3)
pad3 = np.count_nonzero(np.sum(cajas_compradas[filtro_3_malos, :6], axis=1) == 0) / np.count_nonzero(filtro_3_malos)

# Resultado
res = pad3 * p3 / pmas
print(f"La probabilidad de que haya una caja aceptada con 3 integrados malos es:\n{res}")
```

Esto quiere decir que el test tiene baja probabilidad de dar falso positivo. La probabilidad de equivocarnos al aceptar una caja es cercana al 1%. 
