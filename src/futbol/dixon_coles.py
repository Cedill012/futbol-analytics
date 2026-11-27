"""
Modelo Dixon-Coles con decaimiento en el tiempo.

Qué aporta frente a la v2
-------------------------
- Usa el historial completo partido a partido (no una tabla de 17 jornadas):
  cada equipo tiene un ataque y una defensa que se estiman con todos sus
  partidos de la ventana, ajustando por la fuerza del rival.
- Los partidos recientes pesan más: peso = exp(-xi * días de antigüedad).
- Corrige los marcadores bajos (0-0, 1-0, 0-1, 1-1) con el parámetro rho,
  que es justo donde el Poisson simple subestima los empates.

Modelo
------
log(lambda_local)  = mu + local + ataque[local]  - defensa[visita]
log(lambda_visita) = mu +         ataque[visita] - defensa[local]
P(i, j) = tau(i, j) * Poisson(i; lambda_local) * Poisson(j; lambda_visita)

Ajuste en dos pasos (aproximación habitual):
1. Poisson ponderado con penalización L2 (ridge) sobre ataque y defensa,
   con gradiente analítico (rápido).
2. rho por máxima verosimilitud dados los lambdas del paso 1.

Referencia: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

MAX_G = 11


@dataclass
class AjusteDC:
    equipos: dict
    mu: float
    local: float
    ataque: np.ndarray
    defensa: np.ndarray
    rho: float
    escala: float = 1.0  # recalibración al ritmo reciente de goles de la liga

    def lambdas(self, local: str, visita: str, neutral: bool = False):
        def idx(e):
            return self.equipos.get(e)
        il, iv = idx(local), idx(visita)
        at_l = self.ataque[il] if il is not None else 0.0
        de_l = self.defensa[il] if il is not None else 0.0
        at_v = self.ataque[iv] if iv is not None else 0.0
        de_v = self.defensa[iv] if iv is not None else 0.0
        casa = 0.0 if neutral else self.local
        return (self.escala * math.exp(self.mu + casa + at_l - de_v),
                self.escala * math.exp(self.mu + at_v - de_l))

    def matriz(self, local: str, visita: str, neutral: bool = False):
        lam_l, lam_v = self.lambdas(local, visita, neutral)
        return matriz_dc(lam_l, lam_v, self.rho)


def tau(i, j, lam_l, lam_v, rho):
    if i == 0 and j == 0:
        return 1 - lam_l * lam_v * rho
    if i == 0 and j == 1:
        return 1 + lam_l * rho
    if i == 1 and j == 0:
        return 1 + lam_v * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def matriz_dc(lam_l, lam_v, rho):
    pl = [math.exp(-lam_l) * lam_l ** k / math.factorial(k) for k in range(MAX_G)]
    pv = [math.exp(-lam_v) * lam_v ** k / math.factorial(k) for k in range(MAX_G)]
    m = [[pl[i] * pv[j] for j in range(MAX_G)] for i in range(MAX_G)]
    for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
        m[i][j] *= max(tau(i, j, lam_l, lam_v, rho), 1e-9)
    total = sum(map(sum, m))
    return [[x / total for x in fila] for fila in m]


def ajustar(locales, visitas, goles_l, goles_v, pesos, ridge=2.0, x0=None) -> AjusteDC:
    """Ajusta el modelo. Las listas son paralelas (un elemento por partido)."""
    equipos = {e: k for k, e in enumerate(sorted(set(locales) | set(visitas)))}
    n = len(equipos)
    h = np.array([equipos[e] for e in locales])
    a = np.array([equipos[e] for e in visitas])
    gh = np.asarray(goles_l, float)
    ga = np.asarray(goles_v, float)
    w = np.asarray(pesos, float)

    def nll(p):
        mu, casa = p[0], p[1]
        at, de = p[2:2 + n], p[2 + n:]
        lh = mu + casa + at[h] - de[a]
        la = mu + at[a] - de[h]
        eh, ea = np.exp(lh), np.exp(la)
        f = np.sum(w * (eh - gh * lh + ea - ga * la)) + ridge * (at @ at + de @ de)
        rh, ra = w * (eh - gh), w * (ea - ga)
        g = np.empty_like(p)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        g[2:2 + n] = np.bincount(h, rh, n) + np.bincount(a, ra, n) + 2 * ridge * at
        g[2 + n:] = -np.bincount(a, rh, n) - np.bincount(h, ra, n) + 2 * ridge * de
        return f, g

    inicio = np.zeros(2 + 2 * n) if x0 is None or len(x0) != 2 + 2 * n else x0
    if x0 is None:
        inicio[0] = math.log(max(np.average((gh + ga) / 2, weights=w), 0.1))
        inicio[1] = 0.2
    res = minimize(nll, inicio, jac=True, method="L-BFGS-B")
    p = res.x
    mu, casa, at, de = p[0], p[1], p[2:2 + n], p[2 + n:]
    lam_h = np.exp(mu + casa + at[h] - de[a])
    lam_a = np.exp(mu + at[a] - de[h])

    bajos = (gh <= 1) & (ga <= 1)
    lh_b, la_b, gh_b, ga_b, w_b = lam_h[bajos], lam_a[bajos], gh[bajos], ga[bajos], w[bajos]

    def nll_rho(rho):
        t = np.where((gh_b == 0) & (ga_b == 0), 1 - lh_b * la_b * rho,
             np.where((gh_b == 0) & (ga_b == 1), 1 + lh_b * rho,
              np.where((gh_b == 1) & (ga_b == 0), 1 + la_b * rho, 1 - rho)))
        return -np.sum(w_b * np.log(np.clip(t, 1e-9, None)))

    rho = minimize_scalar(nll_rho, bounds=(-0.3, 0.3), method="bounded").x
    ajuste = AjusteDC(equipos, float(mu), float(casa), at, de, float(rho))
    ajuste.parametros = p
    return ajuste


def recalibrar(ajuste: AjusteDC, locales, visitas, goles_l, goles_v, n_recientes: int = 153) -> AjusteDC:
    """Escala los goles esperados para que su promedio en los últimos `n_recientes`
    partidos coincida con los goles reales de la liga en esos partidos.
    El backtest 2014-2026 mostró que esto mejora la predicción de "más de 2.5"."""
    ajuste.escala = 1.0
    ult = slice(-n_recientes, None)
    esperado = np.mean([sum(ajuste.lambdas(l, v)) for l, v in zip(locales[ult], visitas[ult])])
    real = np.mean(np.asarray(goles_l[ult]) + np.asarray(goles_v[ult]))
    ajuste.escala = float(real / esperado)
    return ajuste
