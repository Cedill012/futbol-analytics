"""
Módulo de jugadores: índice de rendimiento y de forma reciente.

Flujo
-----
1. `por_90`            -> convierte estadísticas totales a "por cada 90 minutos".
2. `filtrar_minutos`   -> descarta jugadores con pocos minutos (muestra chica).
3. `percentiles`       -> califica cada métrica 0-100 contra su misma posición.
4. `indice_compuesto`  -> promedio ponderado de percentiles según la posición.
5. `forma_reciente`    -> igual que arriba, pero dando más peso a los últimos partidos.

Formato esperado de datos (ver data/jugadores/plantilla_captura.csv):
una fila por jugador y partido, con columnas: jugador, equipo, posicion, fecha,
minutos y una columna por métrica (goles, tiros_a_puerta, pases_completados, ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTOS_MINIMOS = 450  # ~5 partidos completos

# Pesos por posición: qué importa más en cada una. Ajustables.
PESOS_POR_POSICION: dict[str, dict[str, float]] = {
    "DEL": {"goles": 0.35, "tiros_a_puerta": 0.25, "regates_exitosos": 0.15,
            "pases_clave": 0.15, "duelos_ganados": 0.10},
    "EXT": {"regates_exitosos": 0.30, "pases_clave": 0.25, "goles": 0.20,
            "tiros_a_puerta": 0.15, "recuperaciones": 0.10},
    "MED": {"pases_completados": 0.30, "pases_clave": 0.25, "recuperaciones": 0.25,
            "duelos_ganados": 0.10, "regates_exitosos": 0.10},
    "DEF": {"duelos_ganados": 0.30, "intercepciones": 0.25, "despejes": 0.20,
            "pases_completados": 0.15, "recuperaciones": 0.10},
}

METRICAS = sorted({m for pesos in PESOS_POR_POSICION.values() for m in pesos})


def totales_por_jugador(log: pd.DataFrame) -> pd.DataFrame:
    """Suma el registro partido a partido a totales por jugador."""
    agg = {"minutos": "sum", **{m: "sum" for m in METRICAS if m in log.columns}}
    return (
        log.groupby(["jugador", "equipo", "posicion"], as_index=False)
        .agg(agg)
        .assign(partidos=log.groupby(["jugador", "equipo", "posicion"]).size().values)
    )


def por_90(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas <metrica>_p90."""
    out = df.copy()
    for m in METRICAS:
        if m in out.columns:
            out[f"{m}_p90"] = out[m] / out["minutos"] * 90
    return out


def filtrar_minutos(df: pd.DataFrame, minimo: int = MINUTOS_MINIMOS) -> pd.DataFrame:
    return df[df["minutos"] >= minimo].copy()


def percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Percentil (0-100) de cada métrica p90, comparando solo dentro de la misma posición."""
    out = df.copy()
    for m in METRICAS:
        col = f"{m}_p90"
        if col in out.columns:
            out[f"{m}_pct"] = out.groupby("posicion")[col].rank(pct=True) * 100
    return out


def indice_compuesto(df: pd.DataFrame) -> pd.DataFrame:
    """Índice 0-100: promedio ponderado de percentiles según la posición."""
    out = df.copy()

    def calc(fila):
        pesos = PESOS_POR_POSICION.get(fila["posicion"], {})
        usados = {m: w for m, w in pesos.items() if f"{m}_pct" in fila and pd.notna(fila[f"{m}_pct"])}
        if not usados:
            return np.nan
        total = sum(usados.values())
        return sum(fila[f"{m}_pct"] * w for m, w in usados.items()) / total

    out["indice"] = out.apply(calc, axis=1)
    return out.sort_values("indice", ascending=False)


def forma_reciente(log: pd.DataFrame, ultimos: int = 5, decaimiento: float = 0.75) -> pd.DataFrame:
    """
    Índice de forma: usa los últimos `ultimos` partidos de cada jugador y pesa
    más los recientes (el partido más reciente pesa 1, el anterior `decaimiento`,
    el anterior `decaimiento**2`, etc.).
    """
    log = log.sort_values("fecha")
    filas = []
    for (jugador, equipo, posicion), g in log.groupby(["jugador", "equipo", "posicion"]):
        g = g.tail(ultimos).copy()
        pesos = decaimiento ** np.arange(len(g))[::-1]
        fila = {"jugador": jugador, "equipo": equipo, "posicion": posicion,
                "minutos": float((g["minutos"] * pesos).sum() / pesos.sum() * len(g))}
        for m in METRICAS:
            if m in g.columns:
                fila[m] = float((g[m] * pesos).sum() / pesos.sum() * len(g))
        filas.append(fila)
    ponderado = pd.DataFrame(filas)
    return indice_compuesto(percentiles(por_90(ponderado)))


def ranking(log: pd.DataFrame, minimo_minutos: int = MINUTOS_MINIMOS) -> pd.DataFrame:
    """Ranking de temporada completo: totales -> p90 -> filtro -> percentiles -> índice."""
    tabla = filtrar_minutos(por_90(totales_por_jugador(log)), minimo_minutos)
    return indice_compuesto(percentiles(tabla))
