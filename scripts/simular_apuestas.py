"""Simulación hipotética: ¿cuánto se habría ganado apostando $100 a cada predicción?

No es recomendación de apuesta. Usa los momios de Caliente registrados ANTES de cada partido.
Uso: python scripts/simular_apuestas.py
"""
import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MOMIOS = RAIZ / "data/liga_mx/apertura_2026/momios_caliente_2026-11-25.csv"
VALOR = RAIZ / "predicciones/cuartos_apertura2026_valor_vs_caliente.csv"
RESULTADOS = RAIZ / "predicciones/resultados_cuartos_apertura2026.csv"
SALIDA = RAIZ / "predicciones/simulacion_apuestas_100.csv"
APUESTA = 100

# Predicciones publicadas en el registro del 25/11 (lo que el modelo daba como más probable)
PUBLICADAS = {
    "Toluca vs Atlas": [("1X2", "local"), ("total_1.5", "mas"), ("total_2.5", "menos"), ("ambos_anotan", "no")],
    "Monterrey vs América": [("1X2", "local"), ("total_1.5", "mas"), ("doble_oportunidad", "visitante_empate")],
}


def gano(mercado, seleccion, gl, gv):
    total = gl + gv
    if mercado == "1X2":
        return {"local": gl > gv, "empate": gl == gv, "visitante": gl < gv}[seleccion]
    if mercado == "doble_oportunidad":
        return {"local_empate": gl >= gv, "visitante_empate": gl <= gv}[seleccion]
    if mercado == "ambos_anotan":
        ambos = gl > 0 and gv > 0
        return ambos if seleccion == "si" else not ambos
    if mercado.startswith("total_"):
        linea = float(mercado.split("_")[1])
        return total > linea if seleccion == "mas" else total < linea
    if mercado.startswith("goles_local_"):
        linea = float(mercado.split("_")[2])
        return gl > linea if seleccion == "mas" else gl < linea
    raise ValueError(mercado)


def ganancia(momio, acierto):
    if not acierto:
        return -APUESTA
    return APUESTA * momio / 100 if momio > 0 else APUESTA * 100 / abs(momio)


def main():
    momios = {(r["partido"], r["mercado"], r["seleccion"]): int(r["momio_americano"])
              for r in csv.DictReader(open(MOMIOS, encoding="utf-8"))}
    marcadores = {}
    for r in csv.DictReader(open(RESULTADOS, encoding="utf-8")):
        if r["etapa"] == "ida" and r["goles_local"]:
            marcadores[f'{r["local"]} vs {r["visitante"]}'] = (int(r["goles_local"]), int(r["goles_visitante"]))

    filas = []
    for partido, apuestas in PUBLICADAS.items():
        if partido in marcadores:
            for m, s in apuestas:
                filas.append(("publicadas", partido, m, s, momios[(partido, m, s)]))
    for r in csv.DictReader(open(VALOR, encoding="utf-8")):
        if r["partido"] in marcadores and float(r["valor_esperado"]) > 0:
            filas.append(("valor_positivo", r["partido"], r["mercado"], r["seleccion"], int(r["momio"])))

    salida = []
    for estrategia, partido, m, s, momio in filas:
        gl, gv = marcadores[partido]
        ok = gano(m, s, gl, gv)
        salida.append([estrategia, partido, m, s, momio, f"{gl}-{gv}", ok, round(ganancia(momio, ok), 2)])

    with open(SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["estrategia", "partido", "mercado", "seleccion", "momio", "marcador", "acerto", "ganancia"])
        w.writerows(salida)

    for est in ("publicadas", "valor_positivo"):
        filas_e = [x for x in salida if x[0] == est]
        neto = sum(x[-1] for x in filas_e)
        invertido = APUESTA * len(filas_e)
        print(f"\n== {est} ==")
        for x in filas_e:
            print(f"  {x[1]:22} {x[2]:18} {x[3]:17} {x[4]:>5}  {x[5]}  {'✅' if x[6] else '❌'} {x[7]:>8.2f}")
        print(f"  Apuestas: {len(filas_e)} | Invertido: ${invertido} | Neto: ${neto:.2f} | ROI: {neto / invertido:.1%}")


if __name__ == "__main__":
    main()
