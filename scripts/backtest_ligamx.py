"""
Backtest del modelo de partidos con todos los partidos de Liga MX desde 2012.

Datos: data/liga_mx/historico/MEX.csv (football-data.co.uk, archivo new/MEX.csv).
Cada partido se predice solo con información previa:
  - últimos 17 partidos de cada equipo (mínimo 10),
  - últimos 5 enfrentamientos directos,
  - promedio de goles de los últimos 153 partidos de la liga.

Compara v1 (total fijo de goles), v2 (ataque/defensa con varios encogimientos)
y los momios de cierre de Pinnacle. Escribe predicciones/backtest_ligamx.csv.

Uso: python scripts/backtest_ligamx.py
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO = RAIZ / "data" / "liga_mx" / "historico" / "MEX.csv"
SALIDA = RAIZ / "predicciones" / "backtest_ligamx.csv"

VENTANA_EQUIPO, MINIMO_PARTIDOS, VENTANA_LIGA = 17, 10, 153
MAX_G = 11
FACT = [math.factorial(i) for i in range(MAX_G)]


def pois(lam, k):
    return math.exp(-lam) * lam ** k / FACT[k]


def resumen(lam_l, lam_v, gl, gv):
    """Probabilidades 1X2, más de 2.5 y probabilidad del marcador real."""
    pl, pv = [pois(lam_l, i) for i in range(MAX_G)], [pois(lam_v, j) for j in range(MAX_G)]
    p = [0.0, 0.0, 0.0]
    mas25 = 0.0
    for i in range(MAX_G):
        for j in range(MAX_G):
            q = pl[i] * pv[j]
            p[0 if i > j else 1 if i == j else 2] += q
            if i + j >= 3:
                mas25 += q
    marcador = pl[gl] * pv[gv] if gl < MAX_G and gv < MAX_G else 1e-6
    return p, mas25, marcador


def lambdas_v1(sl, sv, h2h):
    def rating(s):
        return 0.5 * (s["gf"] - s["gc"]) / s["pj"] + 0.5 * ((s["pts"] / s["pj"] - 1.37) / 0.85)
    ventaja = 0.7 * (rating(sl) - rating(sv)) + 0.3 * h2h + 0.30
    return max((2.7 + ventaja) / 2, 0.05), max((2.7 - ventaja) / 2, 0.05)


def lambdas_v2(sl, sv, h2h, mu, k):
    def fuerza(s):
        ataque = (s["gf"] / s["pj"] / mu * s["pj"] + k) / (s["pj"] + k)
        defensa = (s["gc"] / s["pj"] / mu * s["pj"] + k) / (s["pj"] + k)
        return ataque, defensa
    at_l, de_l = fuerza(sl)
    at_v, de_v = fuerza(sv)
    a, b = mu * at_l * de_v, mu * at_v * de_l
    total, dif = a + b, 0.7 * (a - b) + 0.3 * h2h
    return max((total + dif) / 2 + 0.15, 0.05), max((total - dif) / 2 - 0.15, 0.05)


def leer_partidos():
    partidos = []
    with open(ARCHIVO, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r["HG"]:
                continue
            fecha = datetime.strptime(f'{r["Date"]} {r["Time"] or "00:00"}', "%d/%m/%Y %H:%M")
            cuotas = [float(r[c]) if r[c] else None for c in ("PSCH", "PSCD", "PSCA")]
            partidos.append({"fecha": fecha, "local": r["Home"], "visita": r["Away"],
                             "gl": int(r["HG"]), "gv": int(r["AG"]), "cuotas": cuotas})
    partidos.sort(key=lambda x: x["fecha"])
    # Liguilla: los dos equipos ya jugaron 17 partidos en ese torneo (Apertura = jul-dic).
    conteo = defaultdict(int)
    for x in partidos:
        torneo = f'{x["fecha"].year}{"A" if x["fecha"].month >= 7 else "C"}'
        conteo[torneo + x["local"]] += 1
        conteo[torneo + x["visita"]] += 1
        x["liguilla"] = conteo[torneo + x["local"]] > 17 and conteo[torneo + x["visita"]] > 17
    return partidos


def main():
    partidos = leer_partidos()
    modelos = {"v1": None, "v2_k6": 6, "v2_k12": 12, "v2_k24": 24, "v2_k48": 48, "v2_k100": 100}
    met = {m: {g: defaultdict(float) for g in ("todos", "liguilla")} for m in list(modelos) + ["promedio_liga", "pinnacle"]}
    historial, h2h, recientes, mas25_hist = defaultdict(list), defaultdict(list), [], []

    def stats(equipo):
        h = historial[equipo][-VENTANA_EQUIPO:]
        if len(h) < MINIMO_PARTIDOS:
            return None
        return {"pj": len(h), "gf": sum(a for a, _ in h), "gc": sum(b for _, b in h),
                "pts": sum(3 if a > b else 1 if a == b else 0 for a, b in h)}

    for x in partidos:
        sl, sv = stats(x["local"]), stats(x["visita"])
        clave = tuple(sorted((x["local"], x["visita"])))
        previos = h2h[clave][-5:]
        dif_h2h = (sum(p["gl"] - p["gv"] if p["local"] == x["local"] else p["gv"] - p["gl"] for p in previos)
                   / len(previos)) if previos else 0.0
        if sl and sv and len(recientes) >= 150:
            ventana = recientes[-VENTANA_LIGA:]
            mu = sum(ventana) / len(ventana) / 2
            res = 0 if x["gl"] > x["gv"] else 1 if x["gl"] == x["gv"] else 2
            y25 = 1 if x["gl"] + x["gv"] >= 3 else 0
            grupos = ("todos", "liguilla") if x["liguilla"] else ("todos",)
            tasa = sum(mas25_hist[-VENTANA_LIGA:]) / len(mas25_hist[-VENTANA_LIGA:])
            for g in grupos:
                met["promedio_liga"][g]["brier25"] += (tasa - y25) ** 2
                met["promedio_liga"][g]["n"] += 1
            cuotas = x["cuotas"]
            for nombre, k in modelos.items():
                lam = lambdas_v1(sl, sv, dif_h2h) if k is None else lambdas_v2(sl, sv, dif_h2h, mu, k)
                p, p25, pm = resumen(*lam, x["gl"], x["gv"])
                for g in grupos:
                    m = met[nombre][g]
                    m["n"] += 1
                    m["ll_marcador"] += math.log(pm)
                    m["ll_1x2"] += math.log(p[res])
                    m["brier25"] += (p25 - y25) ** 2
                if all(cuotas):
                    m = met[nombre]["todos"]
                    m["n_cuotas"] += 1
                    m["ll_1x2_cuotas"] += math.log(p[res])
                    for o in range(3):
                        if p[o] * cuotas[o] - 1 > 0.05:
                            m["apuestas"] += 1
                            m["ganancia"] += cuotas[o] - 1 if o == res else -1
            if all(cuotas):
                inv = [1 / c for c in cuotas]
                for g in grupos:
                    met["pinnacle"][g]["n"] += 1
                    met["pinnacle"][g]["ll_1x2"] += math.log(inv[res] / sum(inv))
        historial[x["local"]].append((x["gl"], x["gv"]))
        historial[x["visita"]].append((x["gv"], x["gl"]))
        h2h[clave].append(x)
        recientes.append(x["gl"] + x["gv"])
        mas25_hist.append(1 if x["gl"] + x["gv"] >= 3 else 0)

    filas = []
    for nombre, grupos in met.items():
        for g, m in grupos.items():
            n = m["n"]
            if not n:
                continue
            fila = {"modelo": nombre, "grupo": g, "partidos": int(n),
                    "ll_1x2": round(m["ll_1x2"] / n, 4) if m["ll_1x2"] else "",
                    "ll_marcador": round(m["ll_marcador"] / n, 4) if m["ll_marcador"] else "",
                    "brier_mas_2_5": round(m["brier25"] / n, 4) if m["brier25"] else "",
                    "ll_1x2_partidos_con_cuotas": round(m["ll_1x2_cuotas"] / m["n_cuotas"], 4) if m["n_cuotas"] else "",
                    "apuestas_ev_5": int(m["apuestas"]) if m["apuestas"] else "",
                    "roi": f'{m["ganancia"] / m["apuestas"]:.1%}' if m["apuestas"] else ""}
            filas.append(fila)
    with open(SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    lig = [x for x in partidos if x["liguilla"]]
    reg = [x for x in partidos if not x["liguilla"]]
    print(f"Partidos: {len(partidos)} ({len(lig)} de Liguilla)")
    print(f"Goles por partido: Liguilla {sum(x['gl'] + x['gv'] for x in lig) / len(lig):.2f} | "
          f"fase regular {sum(x['gl'] + x['gv'] for x in reg) / len(reg):.2f}")
    for f_ in filas:
        print(f_)


if __name__ == "__main__":
    main()
