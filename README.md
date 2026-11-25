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
- Tabla general y cruces: medios deportivos mexicanos (El Imparcial, Excélsior, TUDN).
- Historial directo: ESPN, TUDN y medios deportivos.
- Momios: Caliente.mx (captura manual, 25/11/2026).

## Autor
Christopher Cedillo · Análisis de datos y automatización
