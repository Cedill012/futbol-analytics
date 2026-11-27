# Vueltas de cuartos · v2 (publicado) vs. v3 Dixon-Coles (en prueba)

Registrado el 27/11/2026, antes de los partidos. Los videos usan la **v2**. La **v3** (Dixon-Coles con decaimiento ξ=0.002, ventana de 3 años, recalibrada al ritmo reciente de goles (escala 1.023); ver `docs/backtest_2012_2026.md`) se registra en paralelo para compararla.

| Partido | v2: L / E / V | v3: L / E / V | v2: más de 2.5 | v3: más de 2.5 | v2: avanza | v3: avanza |
|---|---|---|---|---|---|---|
| Atlas vs Toluca (ida 0-2) | 32 / 23 / 45 | 24 / 23 / 54 | 64% | 64% | Toluca 85% | Toluca 90% |
| América vs Monterrey (ida 0-0) | 47 / 23 / 30 | 55 / 23 / 22 | 62% | 60% | América 70% | América 78% |
| Chivas vs Pachuca (ida 1-2) | — | 46 / 27 / 27 | — | 47% | — | Chivas 46% |
| Cruz Azul vs León (ida 4-0) | — | 69 / 18 / 13 | — | 65% | — | Cruz Azul ~100% |

Nota: la v3 no usa el historial directo de los últimos 5 partidos; la v2 sí, con 30% de peso.
