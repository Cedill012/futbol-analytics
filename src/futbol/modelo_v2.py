"""
Modelo de equipos v2: ataque y defensa por equipo.

Qué cambia respecto a la v1 (modelo_equipos.py)
------------------------------------------------
- v1 usaba el mismo total de goles (2.7) en todos los partidos. Solo movía
  *quién* era favorito, no *cuántos* goles se esperaban.
- v2 le da a cada equipo una fuerza de ataque (goles a favor por partido) y
  una de defensa (goles en contra por partido), relativas al promedio de la
  liga. Así, un partido entre dos equipos goleadores espera más goles que uno
  entre dos equipos cerrados.

Cálculo
-------
1. mu = goles por equipo por partido en la liga (total de goles / partidos / 2).
2. ataque_i  = (GF_i / PJ) / mu      defensa_i = (GC_i / PJ) / mu
3. Encogimiento hacia el promedio (1.0): con 17 partidos hay mucho ruido, así
   que cada valor se mezcla con K_ENCOGIMIENTO "partidos promedio" ficticios.
4. Goles esperados en cancha neutral:  lambda_A = mu * ataque_A * defensa_B
5. Historial directo: la diferencia esperada se mezcla 70/30 con la diferencia
   de goles por partido de los últimos enfrentamientos (como en la v1); el
   total de goles del partido se mantiene.
6. Ventaja de local: +0.15 goles al local y -0.15 al visitante (0.30 de
   diferencia, igual que en la v1).
"""
from __future__ import annotations

import math

import pandas as pd

from futbol.modelo_equipos import MAX_GOLES, PESO_H2H, PESO_TABLA

K_ENCOGIMIENTO = 48.0  # partidos "promedio" ficticios; el backtest 2012-2026 eligió 48 entre 6, 12, 24, 48, 100
LOCAL_MEDIO = 0.15     # goles que gana el local y pierde el visitante
# Se probó bajar los goles en Liguilla (0.89), pero en 429 partidos de fase final
# 2012-2026 el promedio fue 2.66 contra 2.67 en fase regular: no hay diferencia,
# y el factor empeoraba el backtest. Se deja en 1.0.
FACTOR_LIGUILLA = 1.0


def fuerzas_desde_tabla(tabla: pd.DataFrame, k: float = K_ENCOGIMIENTO):
    """Devuelve (mu, {equipo: (ataque, defensa)}) a partir de la tabla completa."""
    partidos = tabla.pj.sum() / 2
    mu = tabla.gf.sum() / partidos / 2
    fuerzas = {}
    for f in tabla.itertuples():
        ataque = (f.gf / f.pj) / mu
        defensa = (f.gc / f.pj) / mu
        ataque = (f.pj * ataque + k) / (f.pj + k)
        defensa = (f.pj * defensa + k) / (f.pj + k)
        fuerzas[f.equipo] = (ataque, defensa)
    return mu, fuerzas


def lambdas_v2(mu, fuerzas, local, visitante, h2h_dif_local=0.0, neutral=False, liguilla=True):
    """Goles esperados (local, visitante). h2h_dif_local: dif. de goles/partido del local en el H2H."""
    at_l, de_l = fuerzas[local]
    at_v, de_v = fuerzas[visitante]
    if liguilla:
        mu = mu * FACTOR_LIGUILLA
    lam_l = mu * at_l * de_v
    lam_v = mu * at_v * de_l
    total = lam_l + lam_v
    dif = PESO_TABLA * (lam_l - lam_v) + PESO_H2H * h2h_dif_local
    lam_l, lam_v = (total + dif) / 2, (total - dif) / 2
    if not neutral:
        lam_l, lam_v = lam_l + LOCAL_MEDIO, lam_v - LOCAL_MEDIO
    return max(lam_l, 0.05), max(lam_v, 0.05)


def _pois(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def matriz(lam_l, lam_v):
    return [[_pois(lam_l, i) * _pois(lam_v, j) for j in range(MAX_GOLES)] for i in range(MAX_GOLES)]


def probabilidades_v2(lam_l, lam_v) -> dict:
    m = matriz(lam_l, lam_v)
    r = range(MAX_GOLES)

    def suma(cond):
        return sum(m[i][j] for i in r for j in r if cond(i, j))

    i, j = max(((i, j) for i in r for j in r), key=lambda x: m[x[0]][x[1]])
    return {"xg_local": lam_l, "xg_visitante": lam_v,
            "gana_local": suma(lambda i, j: i > j), "empate": suma(lambda i, j: i == j),
            "gana_visitante": suma(lambda i, j: i < j),
            "mas_1_5": suma(lambda i, j: i + j >= 2), "mas_2_5": suma(lambda i, j: i + j >= 3),
            "mas_3_5": suma(lambda i, j: i + j >= 4), "ambos_anotan": suma(lambda i, j: i > 0 and j > 0),
            "marcador_probable": f"{i}-{j}", "prob_marcador": m[i][j]}


def prob_avanza_vuelta(lam_mejor_local, lam_rival, global_mejor_tras_ida: int) -> float:
    """El mejor sembrado es local en la vuelta; con global empatado avanza él."""
    m = matriz(lam_mejor_local, lam_rival)
    return sum(m[i][j] for i in range(MAX_GOLES) for j in range(MAX_GOLES)
               if global_mejor_tras_ida + i - j >= 0)
