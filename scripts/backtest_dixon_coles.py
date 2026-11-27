"""
Backtest: Dixon-Coles con decaimiento en el tiempo vs. modelo v2, Liga MX 2014-2026.

Cada semana se vuelve a ajustar Dixon-Coles con los partidos de los últimos
3 años (solo partidos anteriores a la fecha). Se comparan los mismos partidos
para todos los modelos.

Uso: python scripts/backtest_dixon_coles.py
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
from futbol import dixon_coles as dc  # noqa: E402
from backtest_ligamx import leer_partidos, lambdas_v2, resumen  # noqa: E402

INICIO_EVAL = datetime(2014, 7, 1)
VENTANA = timedelta(days=3 * 365)
CONFIGS = {"dc_xi0": 0.0, "dc_xi002": 0.002, "dc_xi004": 0.004, "v3_dc_xi002_recal": 0.002}
RIDGE = 2.0
SALIDA = RAIZ / "predicciones" / "backtest_dixon_coles.csv"


def metricas_matriz(m, gl, gv):
    p = [0.0, 0.0, 0.0]
    mas25 = 0.0
    for i in range(dc.MAX_G):
        for j in range(dc.MAX_G):
            q = m[i][j]
            p[0 if i > j else 1 if i == j else 2] += q
            if i + j >= 3:
                mas25 += q
    marcador = m[gl][gv] if gl < dc.MAX_G and gv < dc.MAX_G else 1e-6
    return p, mas25, marcador


def main():
    partidos = leer_partidos()
    met = {k: defaultdict(float) for k in list(CONFIGS) + ["v2_k48", "pinnacle"]}
    historial, h2h, recientes = defaultdict(list), defaultdict(list), []
    ajustes, ultimo_ajuste = {}, None
    previo = {k: None for k in CONFIGS}

    def stats(e):
        hh = historial[e][-17:]
        if len(hh) < 10:
            return None
        return {"pj": len(hh), "gf": sum(x for x, _ in hh), "gc": sum(y for _, y in hh),
                "pts": sum(3 if x > y else 1 if x == y else 0 for x, y in hh)}

    for idx, x in enumerate(partidos):
        f = x["fecha"]
        sl, sv = stats(x["local"]), stats(x["visita"])
        if f >= INICIO_EVAL and sl and sv and len(recientes) >= 150:
            if ultimo_ajuste is None or (f - ultimo_ajuste).days >= 7:
                ventana = [p for p in partidos[:idx] if f - p["fecha"] <= VENTANA]
                for nombre, xi in CONFIGS.items():
                    pesos = [math.exp(-xi * (f - p["fecha"]).days) for p in ventana]
                    aj = dc.ajustar([p["local"] for p in ventana], [p["visita"] for p in ventana],
                                    [p["gl"] for p in ventana], [p["gv"] for p in ventana], pesos, RIDGE)
                    if nombre.endswith("recal"):
                        dc.recalibrar(aj, [p["local"] for p in ventana], [p["visita"] for p in ventana],
                                      [p["gl"] for p in ventana], [p["gv"] for p in ventana])
                    ajustes[nombre] = aj
                ultimo_ajuste = f
            res = 0 if x["gl"] > x["gv"] else 1 if x["gl"] == x["gv"] else 2
            y25 = 1 if x["gl"] + x["gv"] >= 3 else 0
            cuotas = x["cuotas"]
            preds = {}
            for nombre, aj in ajustes.items():
                preds[nombre] = metricas_matriz(aj.matriz(x["local"], x["visita"]), x["gl"], x["gv"])
            clave = tuple(sorted((x["local"], x["visita"])))
            prev = h2h[clave][-5:]
            dif = (sum(p["gl"] - p["gv"] if p["local"] == x["local"] else p["gv"] - p["gl"] for p in prev)
                   / len(prev)) if prev else 0.0
            ventana_liga = recientes[-153:]
            mu = sum(ventana_liga) / len(ventana_liga) / 2
            preds["v2_k48"] = resumen(*lambdas_v2(sl, sv, dif, mu, 48), x["gl"], x["gv"])
            for nombre, (p, p25, pm) in preds.items():
                m = met[nombre]
                m["n"] += 1
                m["ll_marcador"] += math.log(max(pm, 1e-9))
                m["ll_1x2"] += math.log(p[res])
                m["brier25"] += (p25 - y25) ** 2
                m["p_empate"] += p[1]
                if x["liguilla"]:
                    m["n_lig"] += 1
                    m["ll_1x2_lig"] += math.log(p[res])
                if all(cuotas):
                    m["n_c"] += 1
                    m["ll_1x2_c"] += math.log(p[res])
                    for o in range(3):
                        if p[o] * cuotas[o] - 1 > 0.05:
                            m["apuestas"] += 1
                            m["ganancia"] += cuotas[o] - 1 if o == res else -1
            if all(cuotas):
                inv = [1 / c for c in cuotas]
                met["pinnacle"]["n_c"] += 1
                met["pinnacle"]["ll_1x2_c"] += math.log(inv[res] / sum(inv))
                met["pinnacle"]["p_empate"] += inv[1] / sum(inv)
            met["real"] = met.get("real", defaultdict(float))
            met["real"]["n"] += 1
            met["real"]["empates"] += 1 if res == 1 else 0
        historial[x["local"]].append((x["gl"], x["gv"]))
        historial[x["visita"]].append((x["gv"], x["gl"]))
        h2h[tuple(sorted((x["local"], x["visita"])))].append(x)
        recientes.append(x["gl"] + x["gv"])

    filas = []
    for nombre, m in met.items():
        if nombre == "real":
            continue
        n, nc = m["n"], m["n_c"]
        filas.append({
            "modelo": nombre, "partidos": int(n or nc),
            "ll_1x2": round(m["ll_1x2"] / n, 4) if n else "",
            "ll_1x2_liguilla": round(m["ll_1x2_lig"] / m["n_lig"], 4) if m["n_lig"] else "",
            "ll_marcador": round(m["ll_marcador"] / n, 4) if n else "",
            "brier_mas_2_5": round(m["brier25"] / n, 4) if n else "",
            "ll_1x2_con_cuotas": round(m["ll_1x2_c"] / nc, 4) if nc else "",
            "empate_promedio": round(m["p_empate"] / (n or nc), 3),
            "apuestas_ev_5": int(m["apuestas"]) if m["apuestas"] else "",
            "roi": f'{m["ganancia"] / m["apuestas"]:.1%}' if m["apuestas"] else "",
        })
    with open(SALIDA, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f'Empates reales: {met["real"]["empates"] / met["real"]["n"]:.3f} en {int(met["real"]["n"])} partidos')
    for fila in filas:
        print(fila)


if __name__ == "__main__":
    main()
