"""
Genera las predicciones de cuartos de final del Apertura 2026 y las compara
contra los momios de Caliente.

Uso (desde la raíz del repo):
    python scripts/predecir_cuartos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from futbol import modelo_equipos as me  # noqa: E402

DATOS = RAIZ / "data" / "liga_mx" / "apertura_2026"
SALIDA = RAIZ / "predicciones"


def main() -> None:
    tabla = pd.read_csv(DATOS / "tabla_final.csv")
    h2h = pd.read_csv(DATOS / "h2h_ultimos5.csv")
    cuartos = pd.read_csv(DATOS / "cuartos.csv")
    momios = pd.read_csv(DATOS / "momios_caliente_2026-11-25.csv")
    ratings = me.ratings_desde_tabla(tabla)

    filas_serie, filas_partido, filas_valor = [], [], []
    for s in cuartos.itertuples():
        mejor, rival = s.mejor_sembrado, s.rival
        h2h_serie = h2h[h2h["serie"] == s.serie]
        h2h_mejor = me.h2h_diferencia_por_partido(h2h_serie, mejor)
        ventaja = me.ventaja_neutral(ratings[mejor], ratings[rival], h2h_mejor)

        serie = me.prob_avanza_serie(ventaja)
        filas_serie.append({
            "serie": f"{mejor} vs {rival}",
            "rating_" + "mejor": round(ratings[mejor], 2),
            "rating_rival": round(ratings[rival], 2),
            "h2h_dif_por_partido": round(h2h_mejor, 2),
            "ventaja_neutral": round(ventaja, 2),
            "prob_avanza_" + "mejor": round(serie["avanza_mejor_sembrado"], 3),
            "prob_avanza_rival": round(1 - serie["avanza_mejor_sembrado"], 3),
            "prob_global_empatado": round(serie["global_empatado"], 3),
        })

        for etapa, local, visita, v in [
            ("ida", rival, mejor, -ventaja),
            ("vuelta", mejor, rival, ventaja),
        ]:
            p = me.probabilidades_partido(local, visita, v)
            filas_partido.append({
                "etapa": etapa, "local": local, "visitante": visita,
                "xg_local": round(p.xg_local, 2), "xg_visitante": round(p.xg_visitante, 2),
                "gana_local": round(p.gana_local, 3), "empate": round(p.empate, 3),
                "gana_visitante": round(p.gana_visitante, 3),
                "ambos_anotan": round(p.ambos_anotan, 3), "mas_1_5": round(p.mas_1_5, 3),
                "mas_2_5": round(p.mas_2_5, 3), "menos_3_5": round(p.menos_3_5, 3),
                "anota_local": round(p.anota_local, 3), "anota_visitante": round(p.anota_visitante, 3),
                "marcador_mas_probable": p.marcadores_top[0][0],
            })

            if etapa != "ida":
                continue
            prob_por_mercado = {
                ("1X2", "local"): p.gana_local,
                ("1X2", "empate"): p.empate,
                ("1X2", "visitante"): p.gana_visitante,
                ("ambos_anotan", "si"): p.ambos_anotan,
                ("ambos_anotan", "no"): 1 - p.ambos_anotan,
                ("total_1.5", "mas"): p.mas_1_5,
                ("total_1.5", "menos"): 1 - p.mas_1_5,
                ("total_2.5", "mas"): p.mas_2_5,
                ("total_2.5", "menos"): 1 - p.mas_2_5,
                ("total_3.5", "mas"): 1 - p.menos_3_5,
                ("total_3.5", "menos"): p.menos_3_5,
                ("doble_oportunidad", "local_empate"): p.gana_local + p.empate,
                ("doble_oportunidad", "visitante_empate"): p.gana_visitante + p.empate,
            }
            m = momios[momios["local"] == local]
            for r in m.itertuples():
                prob = prob_por_mercado.get((r.mercado, r.seleccion))
                if prob is None:
                    continue
                filas_valor.append({
                    "partido": r.partido, "mercado": r.mercado, "seleccion": r.seleccion,
                    "momio": r.momio_americano,
                    "prob_casa": round(me.prob_implicita(r.momio_americano), 3),
                    "prob_modelo": round(prob, 3),
                    "valor_esperado": round(me.valor_esperado(prob, r.momio_americano), 3),
                })

    SALIDA.mkdir(exist_ok=True)
    df_serie = pd.DataFrame(filas_serie)
    df_partido = pd.DataFrame(filas_partido)
    df_valor = pd.DataFrame(filas_valor).sort_values("valor_esperado", ascending=False)
    df_serie.to_csv(SALIDA / "cuartos_apertura2026_series.csv", index=False)
    df_partido.to_csv(SALIDA / "cuartos_apertura2026_partidos.csv", index=False)
    df_valor.to_csv(SALIDA / "cuartos_apertura2026_valor_vs_caliente.csv", index=False)

    pd.set_option("display.width", 200)
    print("== Probabilidad de avanzar ==")
    print(df_serie[["serie", "prob_avanza_mejor", "prob_avanza_rival", "prob_global_empatado"]].to_string(index=False))
    print("\n== Partidos ==")
    print(df_partido[["etapa", "local", "visitante", "gana_local", "empate", "gana_visitante", "marcador_mas_probable"]].to_string(index=False))
    print("\n== Top valor esperado vs Caliente (idas) ==")
    print(df_valor.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
