# ⚽ futbol-analytics

Modelos estadísticos para predecir partidos de fútbol y medir el rendimiento de jugadores.
El proyecto avanza por fases: **Liga MX → Premier League → Champions League**.

> Proyecto de portafolio de análisis de datos. Las predicciones se registran **antes** de cada
> partido y después se comparan contra los resultados reales y contra los momios del mercado.

## Fases

| Fase | Estado | Descripción |
|---|---|---|
| 1. Liguilla Liga MX Apertura 2026 | 🟢 En curso | Predicciones en vivo de cuartos, semis y final |
| 2. Premier League | ⚪ Pendiente | Modelo con 10+ temporadas, xG y prueba histórica (backtest) |
| 3. Champions League | ⚪ Pendiente | Partidos entre ligas distintas (ratings Elo) |
| Módulo de jugadores | 🟡 Esqueleto | Índice de rendimiento y forma por posición |

## Primera predicción registrada

**Cuartos de final Apertura 2026**, registrada el 25/11/2026 antes de las idas
([detalle](predicciones/2026-11-25_cuartos_apertura2026.md)):

| Serie | Probabilidad de avanzar |
|---|---|
| Chivas vs Pachuca | Chivas **76.5%** |
| Cruz Azul vs León | Cruz Azul **76.9%** |
| América vs Monterrey | América **67.8%** |
| Atlas vs Toluca | Toluca **65.0%** |

## Seguimiento: ¿acertó el modelo?

| Fecha | Partido | Predicción | Resultado | ¿Acertó? |
|---|---|---|---|---|
| 25/11 | Toluca vs Atlas (ida) | Toluca 61% | 2-0 | ✅ |
| 25/11 | Monterrey vs América (ida) | Monterrey 38% / empate 26% | 0-0 | ❌ |

Detalle y probabilidades actualizadas para la vuelta:
[evaluación del 26/11](predicciones/2026-11-26_evaluacion_idas_miercoles.md).

## Backtest: ¿le gana el modelo al mercado?

Probé el modelo con **4,812 partidos de Liga MX (2012–2026)**, prediciendo cada uno solo con la información previa ([detalle](docs/backtest_2012_2026.md)).

| Modelo | Log-verosimilitud 1X2 (más alto = mejor) | Brier "más de 2.5" (más bajo = mejor) | ROI apostando con EV > 5% |
|---|---|---|---|
| v1 (total de goles fijo) | −1.060 | 0.2500 | −8.7% |
| **v2 (ataque y defensa, k=48)** | **−1.057** | **0.2487** | −9.3% |
| Momios de cierre de Pinnacle | **−1.023** | — | — |

**Conclusión honesta:** la v2 mejora un poco a la v1 y es la que se usa ahora. Ninguna de las dos le gana al mercado: cuando el modelo "ve valor", lo más probable es que se equivoque el modelo. Por eso las predicciones se publican como estadística descriptiva, no como recomendación de apuesta.

## Metodología (modelo de equipos v0.1)

1. **Rating por equipo** (goles de ventaja por partido contra un rival promedio):
   50% diferencia de goles por partido + 50% puntos por partido convertidos a goles.
2. **Mezcla con historial directo:** 70% rating de temporada + 30% diferencia de goles promedio
   en los últimos 5 enfrentamientos.
3. **Goles esperados por equipo:** total de 2.7 goles por partido repartido según la ventaja,
   más 0.3 goles por jugar de local.
4. **Distribución de Poisson** para cada marcador → probabilidades de 1X2, ambos anotan, totales, etc.
5. **Series de ida y vuelta:** suma de las dos distribuciones; con global empatado avanza el mejor sembrado.
6. **Comparación con el mercado:** probabilidad implícita de los momios y valor esperado.

### Limitaciones conocidas
- El total de goles es el mismo para todos los partidos (no distingue partidos abiertos de cerrados).
- Muestra pequeña: 17 partidos de temporada y 5 de historial.
- No considera lesiones, alineaciones ni forma reciente (es estadística pura, a propósito).

## Módulo de jugadores

`src/futbol/jugadores.py` calcula un índice 0-100 por jugador:
estadísticas **por 90 minutos** → **percentiles dentro de su posición** → **promedio ponderado**
según lo que importa en cada posición (DEL, EXT, MED, DEF). `forma_reciente()` da más peso a los
últimos 5 partidos. Formato de captura: `data/jugadores/plantilla_captura.csv`.

## Estructura

```
futbol-analytics/
├── data/
│   ├── liga_mx/apertura_2026/   # tabla, historial directo, cruces y momios
│   └── jugadores/               # plantilla de captura de estadísticas
├── src/futbol/
│   ├── modelo_equipos.py        # ratings, Poisson, series, momios
│   └── jugadores.py             # p90, percentiles, índice, forma
├── scripts/predecir_cuartos.py  # genera las predicciones en /predicciones
├── predicciones/                # registro con fecha + resultados reales
├── notebooks/                   # análisis exploratorio y gráficas
└── tests/                       # pruebas automáticas
```

## Cómo correrlo

```bash
pip install -r requirements.txt
python scripts/predecir_cuartos.py
pytest
```

## Fuentes de datos
- Resultados históricos y momios de cierre: [football-data.co.uk](https://www.football-data.co.uk/mexico.php) (`new/MEX.csv`)
- Tabla general y cruces: medios deportivos mexicanos (El Imparcial, Excélsior, TUDN).
- Historial directo: ESPN, TUDN y medios deportivos.
- Momios: Caliente.mx (captura manual, 25/11/2026).

## Autor
Christopher Cedillo · Análisis de datos y automatización
