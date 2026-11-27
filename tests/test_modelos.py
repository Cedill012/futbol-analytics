import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from futbol import jugadores as jg  # noqa: E402
from futbol import modelo_equipos as me  # noqa: E402


def test_probabilidades_suman_uno():
    p = me.probabilidades_partido("A", "B", 0.4)
    assert p.gana_local + p.empate + p.gana_visitante == pytest.approx(1, abs=1e-6)


def test_equipos_iguales_cancha_neutral_simetrico():
    # Sin ventaja: quitar localía para comprobar simetría
    lam_l, lam_v = me.lambdas(0.0)
    assert lam_l == pytest.approx(lam_v)


def test_serie_equipos_iguales_favorece_mejor_sembrado():
    # Con equipos iguales, el desempate por tabla da > 50% al mejor sembrado
    r = me.prob_avanza_serie(0.0)
    assert r["avanza_mejor_sembrado"] > 0.5


def test_momios():
    assert me.americano_a_decimal(150) == pytest.approx(2.5)
    assert me.americano_a_decimal(-200) == pytest.approx(1.5)


def test_indice_jugadores_rango():
    log = pd.DataFrame([
        {"jugador": f"J{i}", "equipo": "X", "posicion": "DEL", "fecha": "2026-01-0" + str(k),
         "minutos": 90, "goles": i % 3, "tiros_a_puerta": i, "regates_exitosos": 1,
         "pases_clave": 1, "duelos_ganados": 2}
        for i in range(1, 6) for k in range(1, 7)
    ])
    r = jg.ranking(log)
    assert r["indice"].between(0, 100).all()
    assert r.iloc[0]["jugador"] == "J5"


def test_v2_partido_abierto_espera_mas_goles():
    import pandas as pd
    from futbol import modelo_v2 as v2
    tabla = pd.DataFrame({"equipo": ["Goleador", "Muralla", "Promedio"], "pj": [17, 17, 17],
                          "gf": [40, 15, 25], "gc": [35, 12, 25]})
    mu, F = v2.fuerzas_desde_tabla(tabla)
    abierto = sum(v2.lambdas_v2(mu, F, "Goleador", "Promedio", neutral=True))
    cerrado = sum(v2.lambdas_v2(mu, F, "Muralla", "Promedio", neutral=True))
    assert abierto > cerrado
    p = v2.probabilidades_v2(1.4, 1.1)
    assert abs(p["gana_local"] + p["empate"] + p["gana_visitante"] - 1) < 1e-6


def test_dixon_coles_equipo_fuerte_favorito():
    from futbol import dixon_coles as dc
    loc, vis, gl, gv = [], [], [], []
    for _ in range(20):
        loc += ["Fuerte", "Debil"]; vis += ["Debil", "Fuerte"]; gl += [3, 0]; gv += [0, 2]
    aj = dc.ajustar(loc, vis, gl, gv, [1.0] * len(loc), ridge=0.5)
    m = aj.matriz("Fuerte", "Debil")
    gana = sum(m[i][j] for i in range(dc.MAX_G) for j in range(dc.MAX_G) if i > j)
    assert gana > 0.7
    assert abs(sum(map(sum, m)) - 1) < 1e-9
