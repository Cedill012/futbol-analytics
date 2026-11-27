"""
Recalcula la probabilidad de avanzar de cada serie con los resultados de la ida
ya registrados en predicciones/resultados_cuartos_apertura2026.csv.

Uso: python scripts/actualizar_tras_ida.py
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from futbol import modelo_equipos as me  # noqa: E402

DATOS = RAIZ / "data" / "liga_mx" / "apertura_2026"
tabla = pd.read_csv(DATOS / "tabla_final.csv")
h2h = pd.read_csv(DATOS / "h2h_ultimos5.csv")
cuartos = pd.read_csv(DATOS / "cuartos.csv")
res = pd.read_csv(RAIZ / "predicciones" / "resultados_cuartos_apertura2026.csv")
ratings = me.ratings_desde_tabla(tabla)

filas = []
for s in cuartos.itertuples():
    mejor, rival = s.mejor_sembrado, s.rival
    ida = res[(res.etapa == "ida") & (res.local == rival) & (res.visitante == mejor)]
    if ida.empty or pd.isna(ida.iloc[0].goles_local):
        continue
    g = int(ida.iloc[0].goles_visitante) - int(ida.iloc[0].goles_local)
    v = me.ventaja_neutral(ratings[mejor], ratings[rival],
                           me.h2h_diferencia_por_partido(h2h[h2h.serie == s.serie], mejor))
    p = me.prob_avanza_tras_ida(v, g)
    filas.append({"serie": f"{mejor} vs {rival}",
                  "ida": f"{rival} {int(ida.iloc[0].goles_local)}-{int(ida.iloc[0].goles_visitante)} {mejor}",
                  "prob_avanza_mejor": round(p, 3), "prob_avanza_rival": round(1 - p, 3)})

out = pd.DataFrame(filas)
out.to_csv(RAIZ / "predicciones" / "cuartos_apertura2026_series_tras_ida.csv", index=False)
print(out.to_string(index=False))
