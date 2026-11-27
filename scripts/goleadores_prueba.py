"""
PRUEBA (no se publica): probabilidad de que cada jugador anote en un partido.

Datos: tabla de goleo individual de la fase regular (ligamx.net), goles y minutos.
Modelo:
  1. Goles esperados del equipo: modelo v2.
  2. Ritmo del jugador cuando está en la cancha, encogido hacia 0.1 goles/90
     (se le suman 0.5 goles en 450 minutos ficticios para no exagerar muestras chicas).
  3. Parte de los goles del equipo que le toca = ritmo del jugador / ritmo del equipo.
  4. Minutos esperados según su uso en el torneo (80, 60 o 30).
  5. P(anota al menos 1) = 1 - exp(-goles esperados del jugador).
Limitaciones: no sabe alineaciones, lesiones ni quién tira los penales.

Uso: python scripts/goleadores_prueba.py "América" "Monterrey" [más pares local visita]
"""
import math
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from futbol import modelo_equipos as v1, modelo_v2 as v2  # noqa: E402

D = RAIZ / "data" / "liga_mx" / "apertura_2026"
MIN_TEMPORADA = 17 * 90


def minutos_esperados(min_jugados):
    return 80 if min_jugados >= 900 else 60 if min_jugados >= 500 else 30


def main(pares):
    tabla = pd.read_csv(D / "tabla_completa_gf_gc.csv")
    goleo = pd.read_csv(D / "goleo_individual_liguilla.csv")
    h2h = pd.read_csv(D / "h2h_ultimos5.csv")
    cuartos = pd.read_csv(D / "cuartos.csv")
    mu, fuerzas = v2.fuerzas_desde_tabla(tabla)
    gf = dict(zip(tabla.equipo, tabla.gf))
    filas = []
    for local, visita in pares:
        serie = cuartos[(cuartos.mejor_sembrado == local) | (cuartos.rival == local)].iloc[0].serie
        dif = v1.h2h_diferencia_por_partido(h2h[h2h.serie == serie], local)
        lam = dict(zip((local, visita), v2.lambdas_v2(mu, fuerzas, local, visita, dif)))
        for equipo in (local, visita):
            ritmo_equipo = gf[equipo] / MIN_TEMPORADA
            for j in goleo[goleo.equipo == equipo].itertuples():
                ritmo = (j.goles + 0.5) / (j.minutos + 450)
                lam_j = lam[equipo] * (ritmo / ritmo_equipo) * minutos_esperados(j.minutos) / 90
                filas.append({"partido": f"{local} vs {visita}", "equipo": equipo, "jugador": j.jugador,
                              "goles": j.goles, "minutos": j.minutos,
                              "goles_esperados_equipo": round(lam[equipo], 2),
                              "prob_anota": round(1 - math.exp(-lam_j), 3)})
    out = pd.DataFrame(filas).sort_values(["partido", "equipo", "prob_anota"], ascending=[True, True, False])
    return out


if __name__ == "__main__":
    args = sys.argv[1:] or ["Atlas", "Toluca", "América", "Monterrey"]
    pares = list(zip(args[::2], args[1::2]))
    res = main(pares)
    res.to_csv(RAIZ / "predicciones" / "pruebas" / "goleadores_2026-11-28.csv", index=False)
    print(res.groupby("equipo").head(4).to_string(index=False))
