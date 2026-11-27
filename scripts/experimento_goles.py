"""
Experimento: ¿cómo predecir mejor el total de goles (más de 2.5)?

Se comparan en los mismos partidos (2014-2026), con Brier "más de 2.5" (más bajo = mejor):
  A. Promedio de la liga (últimos 153 partidos)
  B. Dixon-Coles v3 (xi=0.002)
  C. Dixon-Coles recalibrado: los goles esperados se escalan para que su promedio
     coincida con el promedio reciente de la liga (tendencias del torneo).
  D. Goles implícitos en el 1X2 de Pinnacle: se buscan los lambdas que reproducen
     las probabilidades de cierre del mercado y con ellos se calcula más de 2.5.
  E. Mezcla 50/50 de B y D.
Uso: python scripts/experimento_goles.py
"""
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
from futbol import dixon_coles as dc  # noqa: E402
from backtest_ligamx import leer_partidos  # noqa: E402

R = range(dc.MAX_G)


def mas25(m):
    return sum(m[i][j] for i in R for j in R if i + j >= 3)


def uno_x_dos(m):
    return (sum(m[i][j] for i in R for j in R if i > j), sum(m[i][i] for i in R))


def lambdas_mercado(pl, pe, rho):
    def f(x):
        a, b = np.exp(x)
        m = dc.matriz_dc(a, b, rho)
        l, e = uno_x_dos(m)
        return [l - pl, e - pe]
    s = least_squares(f, [0.3, 0.1], bounds=([-2, -2], [1.5, 1.5]))
    return np.exp(s.x)


def main():
    P = leer_partidos()
    inicio = datetime(2014, 7, 1)
    B = defaultdict(float)
    n = 0
    aj, ult = None, None
    recientes = []
    for k, x in enumerate(P):
        f = x["fecha"]
        if f >= inicio and len(recientes) >= 150 and all(x["cuotas"]):
            if ult is None or (f - ult).days >= 7:
                V = [p for p in P[:k] if f - p["fecha"] <= timedelta(days=1095)]
                w = [math.exp(-0.002 * (f - p["fecha"]).days) for p in V]
                aj = dc.ajustar([p["local"] for p in V], [p["visita"] for p in V],
                                [p["gl"] for p in V], [p["gv"] for p in V], w, 2.0)
                media_dc = np.mean([sum(aj.lambdas(p["local"], p["visita"])) for p in V[-153:]])
                ult = f
            y = 1 if x["gl"] + x["gv"] >= 3 else 0
            liga = sum(1 for t in recientes[-153:] if t >= 3) / 153
            media_liga = sum(recientes[-153:]) / 153
            lh, la = aj.lambdas(x["local"], x["visita"])
            pB = mas25(dc.matriz_dc(lh, la, aj.rho))
            esc = media_liga / media_dc
            pC = mas25(dc.matriz_dc(lh * esc, la * esc, aj.rho))
            inv = [1 / c for c in x["cuotas"]]
            s = sum(inv)
            mh, ma = lambdas_mercado(inv[0] / s, inv[1] / s, aj.rho)
            pD = mas25(dc.matriz_dc(mh, ma, aj.rho))
            for nombre, p in (("A_liga", liga), ("B_dc", pB), ("C_dc_recalibrado", pC),
                              ("D_mercado_1x2", pD), ("E_mezcla_B_D", (pB + pD) / 2), ("F_mezcla_C_D", (pC + pD) / 2)):
                B[nombre] += (p - y) ** 2
                B["acierto_" + nombre] += 1 if (p >= 0.5) == (y == 1) else 0
            n += 1
        recientes.append(x["gl"] + x["gv"])
    print(f"Partidos: {n}")
    for k_, v in sorted(((k, v) for k, v in B.items() if not k.startswith("acierto_")), key=lambda t: t[1]):
        print(f"{k_:20} Brier {v / n:.4f}   acierta el lado (más/menos 2.5): {B['acierto_' + k_] / n:.1%}")


if __name__ == "__main__":
    main()
