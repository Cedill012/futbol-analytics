"""
Modelo de equipos: probabilidades de partido y de serie (ida y vuelta) con Poisson.

Idea general
------------
1. Cada equipo recibe un *rating* en "goles de ventaja por partido frente a un
   rival promedio", calculado con la tabla de la fase regular.
2. El rating se mezcla con el historial directo (últimos enfrentamientos).
3. La diferencia de ratings + ventaja de local define cuántos goles se esperan
   de cada lado (lambda) y con una distribución de Poisson salen las
   probabilidades de cada marcador.

Supuestos (versión 0.1)
-----------------------
- Mismo total de goles esperado en todos los partidos (TOTAL_GOLES). Es la
  limitación principal: no distingue partidos abiertos de cerrados.
- Goles de cada equipo independientes entre sí (Poisson simple, sin Dixon-Coles).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# Parámetros del modelo ---------------------------------------------------------
PPG_PROMEDIO_LIGA = 1.37   # puntos por partido de un equipo promedio
GOLES_POR_PUNTO = 0.85     # cuántos puntos por partido vale 1 gol de diferencia
PESO_TABLA = 0.70          # peso de la temporada vs. historial directo
PESO_H2H = 0.30
VENTAJA_LOCAL = 0.30       # goles extra por jugar en casa
TOTAL_GOLES = 2.70         # goles esperados por partido (promedio de liga)
MAX_GOLES = 12             # límite para sumar la distribución


# Ratings ----------------------------------------------------------------------
def rating_equipo(pts: int, dg: int, pj: int) -> float:
    """Rating en goles/partido: 50% diferencia de goles, 50% puntos por partido."""
    dg_por_partido = dg / pj
    ppg = pts / pj
    dg_implicita_por_puntos = (ppg - PPG_PROMEDIO_LIGA) / GOLES_POR_PUNTO
    return 0.5 * dg_por_partido + 0.5 * dg_implicita_por_puntos


def ratings_desde_tabla(tabla: pd.DataFrame) -> dict[str, float]:
    return {
        fila.equipo: rating_equipo(fila.pts, fila.dg, fila.pj)
        for fila in tabla.itertuples()
    }


def h2h_diferencia_por_partido(h2h: pd.DataFrame, equipo: str) -> float:
    """Diferencia de goles promedio de `equipo` en los enfrentamientos dados."""
    dif = []
    for fila in h2h.itertuples():
        if fila.local == equipo:
            dif.append(fila.goles_local - fila.goles_visitante)
        elif fila.visitante == equipo:
            dif.append(fila.goles_visitante - fila.goles_local)
    if not dif:
        return 0.0
    return sum(dif) / len(dif)


def ventaja_neutral(rating_a: float, rating_b: float, h2h_a: float = 0.0) -> float:
    """Goles de ventaja esperados de A sobre B en cancha neutral."""
    return PESO_TABLA * (rating_a - rating_b) + PESO_H2H * h2h_a


# Partido ----------------------------------------------------------------------
def _pois(lam: float, k: int) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def lambdas(ventaja_local_total: float, total: float = TOTAL_GOLES) -> tuple[float, float]:
    """Goles esperados (local, visitante) a partir de la ventaja del local."""
    lam_local = max((total + ventaja_local_total) / 2, 0.05)
    lam_visita = max((total - ventaja_local_total) / 2, 0.05)
    return lam_local, lam_visita


def matriz_marcadores(lam_local: float, lam_visita: float) -> list[list[float]]:
    return [
        [_pois(lam_local, i) * _pois(lam_visita, j) for j in range(MAX_GOLES)]
        for i in range(MAX_GOLES)
    ]


@dataclass
class ProbPartido:
    local: str
    visitante: str
    xg_local: float
    xg_visitante: float
    gana_local: float
    empate: float
    gana_visitante: float
    ambos_anotan: float
    mas_1_5: float
    mas_2_5: float
    menos_3_5: float
    anota_local: float
    anota_visitante: float
    marcadores_top: list[tuple[str, float]]


def probabilidades_partido(local: str, visitante: str, ventaja_neutral_local: float) -> ProbPartido:
    """Probabilidades de un partido. `ventaja_neutral_local` = ventaja del local sin contar la localía."""
    lam_l, lam_v = lambdas(ventaja_neutral_local + VENTAJA_LOCAL)
    m = matriz_marcadores(lam_l, lam_v)
    rango = range(MAX_GOLES)

    def suma(cond):
        return sum(m[i][j] for i in rango for j in rango if cond(i, j))

    top = sorted(
        ((f"{i}-{j}", m[i][j]) for i in range(6) for j in range(6)),
        key=lambda x: x[1],
        reverse=True,
    )[:3]
    return ProbPartido(
        local=local,
        visitante=visitante,
        xg_local=lam_l,
        xg_visitante=lam_v,
        gana_local=suma(lambda i, j: i > j),
        empate=suma(lambda i, j: i == j),
        gana_visitante=suma(lambda i, j: i < j),
        ambos_anotan=suma(lambda i, j: i > 0 and j > 0),
        mas_1_5=suma(lambda i, j: i + j >= 2),
        mas_2_5=suma(lambda i, j: i + j >= 3),
        menos_3_5=suma(lambda i, j: i + j <= 3),
        anota_local=1 - math.exp(-lam_l),
        anota_visitante=1 - math.exp(-lam_v),
        marcadores_top=top,
    )


def _distribucion_diferencia(lam_a: float, lam_b: float) -> dict[int, float]:
    d: dict[int, float] = {}
    for i in range(MAX_GOLES):
        for j in range(MAX_GOLES):
            d[i - j] = d.get(i - j, 0.0) + _pois(lam_a, i) * _pois(lam_b, j)
    return d


def prob_avanza_serie(ventaja_neutral_mejor: float, empate_global_favorece_mejor: bool = True) -> dict[str, float]:
    """
    Probabilidad de que el mejor sembrado avance en una serie ida y vuelta.
    El mejor sembrado juega la ida de visitante y la vuelta en casa.
    """
    # Ida: mejor sembrado de visitante
    lam_l, lam_v = lambdas(-ventaja_neutral_mejor + VENTAJA_LOCAL)
    ida = _distribucion_diferencia(lam_v, lam_l)          # diferencia desde el mejor sembrado
    # Vuelta: mejor sembrado de local
    lam_l2, lam_v2 = lambdas(ventaja_neutral_mejor + VENTAJA_LOCAL)
    vuelta = _distribucion_diferencia(lam_l2, lam_v2)

    serie: dict[int, float] = {}
    for d1, p1 in ida.items():
        for d2, p2 in vuelta.items():
            serie[d1 + d2] = serie.get(d1 + d2, 0.0) + p1 * p2

    gana = sum(p for d, p in serie.items() if d > 0)
    empate = serie.get(0, 0.0)
    avanza = gana + (empate if empate_global_favorece_mejor else empate / 2)
    return {"avanza_mejor_sembrado": avanza, "global_empatado": empate}


# Momios -----------------------------------------------------------------------
def americano_a_decimal(momio: int) -> float:
    return 1 + momio / 100 if momio > 0 else 1 + 100 / abs(momio)


def prob_implicita(momio: int) -> float:
    return 1 / americano_a_decimal(momio)


def valor_esperado(prob_modelo: float, momio: int) -> float:
    """Ganancia esperada por cada 1 apostado (0.05 = +5%)."""
    return prob_modelo * americano_a_decimal(momio) - 1


def prob_avanza_tras_ida(ventaja_neutral_mejor: float, global_mejor_tras_ida: int) -> float:
    """
    Probabilidad de que el mejor sembrado avance, ya jugada la ida.
    `global_mejor_tras_ida`: diferencia de goles del mejor sembrado después de la ida
    (ej. -2 si perdió 2-0). La vuelta se juega en casa del mejor sembrado y el empate
    global lo favorece.
    """
    lam_l, lam_v = lambdas(ventaja_neutral_mejor + VENTAJA_LOCAL)
    m = matriz_marcadores(lam_l, lam_v)
    return sum(
        m[i][j]
        for i in range(MAX_GOLES)
        for j in range(MAX_GOLES)
        if global_mejor_tras_ida + (i - j) >= 0
    )
