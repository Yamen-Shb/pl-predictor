"""
poisson.py — Score-probability engine for soccer match prediction.

Supports three score-matrix modes (mutually exclusive; bivariate takes priority):
  1. Independent Poisson                  (bivariate_l3=0, dixon_coles_rho=0)
  2. Independent Poisson + Dixon-Coles    (bivariate_l3=0, dixon_coles_rho != 0)
  3. Bivariate Poisson                    (bivariate_l3 > 0)

Dixon–Coles reference:
  Dixon, M. J., & Coles, S. G. (1997).
  "Modelling Association Football Scores and Inefficiencies in the Football
  Betting Market." Journal of the Royal Statistical Society: Series C
  (Applied Statistics), 46(2), 265–280.
  https://doi.org/10.1111/1467-9876.00065

Bivariate Poisson reference:
  Karlis, D., & Ntzoufras, I. (2003).
  "Analysis of sports data by using bivariate Poisson models."
  Journal of the Royal Statistical Society: Series D (The Statistician),
  52(3), 381–393.
  https://doi.org/10.1111/1467-9884.00366

MAX_GOALS (default 7): upper truncation ceiling for all score matrices.
  Goals beyond this are absorbed into the last bin via renormalization.
  For very high λ (> ~4) a meaningful fraction of mass is truncated —
  the normalization step redistributes it proportionally across [0, MAX_GOALS],
  slightly inflating all probabilities. Raise MAX_GOALS if you need accuracy
  for high-scoring leagues or specific over/under markets.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from typing import List, Dict

MAX_GOALS: int = 7


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_rate(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


def _validate_rho(rho: float) -> float:
    rho = float(rho)
    if not np.isfinite(rho):
        raise ValueError(f"dixon_coles_rho must be finite, got {rho!r}")
    return rho


# ---------------------------------------------------------------------------
# Truncated PMF
# ---------------------------------------------------------------------------

def truncated_pmf(lam: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Return a renormalized Poisson PMF over [0, max_goals].

    Truncation: probability mass above max_goals is redistributed
    proportionally across all bins (via division by pmf.sum()).

    Fallback: if all computed probabilities are zero (extremely large λ
    relative to max_goals), all mass is placed at the ceiling bin
    (max_goals) rather than at 0, which is the more plausible score.

    Args:
        lam:       Poisson rate (>= 0, finite).
        max_goals: Upper bin, inclusive.

    Returns:
        1-D array of length (max_goals + 1) summing to 1.
    """
    lam = _validate_rate(lam, "lam")

    ks = np.arange(0, max_goals + 1)
    pmf = poisson.pmf(ks, lam)
    total = pmf.sum()

    if total <= 0:
        # lam so large that all bins underflow to 0 — put mass at ceiling
        fallback = np.zeros(max_goals + 1)
        fallback[max_goals] = 1.0
        return fallback

    return pmf / total


# ---------------------------------------------------------------------------
# Bivariate Poisson helpers
# ---------------------------------------------------------------------------

def _bivariate_poisson_pmf(x: int, y: int,
                            l1: float, l2: float, l3: float) -> float:
    """
    P(Home = x, Away = y) under Bivariate Poisson(l1, l2, l3).

    Decomposition:
        Home goals  ~ Poisson(l1 + l3)
        Away goals  ~ Poisson(l2 + l3)
        Cov(H, A)   = l3   (shared 'match intensity' component)

    Joint PMF (closed form via convolution over the shared component k):
        P(X=x, Y=y) = exp(-(l1+l2+l3)) * sum_{k=0}^{min(x,y)}
                        [ Poisson(k; l3) * Poisson(x-k; l1) * Poisson(y-k; l2) ]

    Complexity: O(min(x, y)) per cell — negligible for MAX_GOALS <= 10.
    """
    k_max = min(x, y)
    total = 0.0
    for k in range(k_max + 1):          # O(k) inner sum
        total += (
            poisson.pmf(k,     l3) *
            poisson.pmf(x - k, l1) *
            poisson.pmf(y - k, l2)
        )
    return total


def _bivariate_poisson_matrix(l1: float, l2: float,
                               l3: float, max_goals: int) -> np.ndarray:
    """
    Build a (max_goals+1) x (max_goals+1) score matrix under Bivariate Poisson.

    Args:
        l1: Independent home rate (your XGBoost home-goals prediction).
        l2: Independent away rate (your XGBoost away-goals prediction).
        l3: Shared covariance term. Fit via MLE on a validation set;
            typical values 0.05–0.20. Higher l3 → more diagonal mass → more draws.
        max_goals: Truncation ceiling (inclusive).
    """
    n = max_goals + 1
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = _bivariate_poisson_pmf(i, j, l1, l2, l3)
    return matrix


# ---------------------------------------------------------------------------
# Main score matrix builder
# ---------------------------------------------------------------------------

def poisson_matrix(
    home_pred: float,
    away_pred: float,
    max_goals: int = MAX_GOALS,
    dixon_coles_rho: float = 0.0,
    bivariate_l3: float = 0.0,
) -> np.ndarray:
    """
    Build a (max_goals+1) x (max_goals+1) score-probability matrix.

    Layout: matrix[i, j] = P(home scores i, away scores j).

    Mode selection (bivariate takes priority if both are set):
      bivariate_l3 > 0  → Bivariate Poisson (structural goal correlation).
      dixon_coles_rho   → Independent Poisson + Dixon-Coles low-score patch.
      both == 0         → Plain independent Poisson.

    Dixon–Coles tau correction (applied only in independent mode):
      Adjusts the four low-score cells to correct systematic biases.
      The tau factors are:
        tau(0,0) = 1 - rho * lam_h * lam_a
        tau(0,1) = 1 + rho * lam_h
        tau(1,0) = 1 + rho * lam_a
        tau(1,1) = 1 - rho

      **Sign of rho matters:**
        Positive rho suppresses 0-0 and 1-1 (boosts 1-0 and 0-1)
          → decreases draw probability, increases decisive results.
        Negative rho boosts 0-0 and 1-1 (suppresses 1-0 and 0-1)
          → increases draw probability.
      Fit rho via MLE on your validation set — let the data decide the sign.
      Typical fitted values fall in the range [-0.15, +0.15].
      Reference: Dixon & Coles (1997), Section 2.

    Args:
        home_pred:        Predicted home goals (lambda from your model).
        away_pred:        Predicted away goals (lambda from your model).
        max_goals:        Score truncation ceiling (inclusive). Default 7.
        dixon_coles_rho:  DC correction parameter. 0 = disabled.
        bivariate_l3:     Bivariate Poisson covariance term. 0 = disabled.

    Returns:
        2-D float array of shape (max_goals+1, max_goals+1), summing to 1.
    """
    home_pred = _validate_rate(home_pred, "home_pred")
    away_pred = _validate_rate(away_pred, "away_pred")
    dixon_coles_rho = _validate_rho(dixon_coles_rho)
    bivariate_l3 = _validate_rate(bivariate_l3, "bivariate_l3")

    if bivariate_l3 > 0.0:
        score_matrix = _bivariate_poisson_matrix(
            home_pred, away_pred, bivariate_l3, max_goals
        )

    else:
        home_probs = truncated_pmf(home_pred, max_goals)
        away_probs = truncated_pmf(away_pred, max_goals)
        score_matrix = np.outer(home_probs, away_probs)

        if dixon_coles_rho != 0.0:
            tau = np.ones_like(score_matrix)
            tau[0, 0] = 1.0 - dixon_coles_rho * home_pred * away_pred
            tau[0, 1] = 1.0 + dixon_coles_rho * home_pred
            tau[1, 0] = 1.0 + dixon_coles_rho * away_pred
            tau[1, 1] = 1.0 - dixon_coles_rho
            tau = np.clip(tau, 0.0, None)
            score_matrix = score_matrix * tau

    total = score_matrix.sum()
    if total > 0:
        score_matrix = score_matrix / total

    return score_matrix


# ---------------------------------------------------------------------------
# Outcome probabilities
# ---------------------------------------------------------------------------

def match_outcome_probabilities(
    score_matrix: np.ndarray,
) -> tuple[float, float, float]:
    """
    Derive W/D/L probabilities from a score matrix.

    score_matrix[i, j] = P(home scores i, away scores j)

    Home win:  i > j  → lower triangle  (np.tril, k=-1)
    Draw:      i == j → main diagonal
    Away win:  j > i  → upper triangle  (np.triu, k=+1)

    Returns:
        (home_win_prob, draw_prob, away_win_prob)  — sum to ~1.
    """
    home_win_prob = float(np.sum(np.tril(score_matrix, -1)))
    draw_prob     = float(np.sum(np.diag(score_matrix)))
    away_win_prob = float(np.sum(np.triu(score_matrix, 1)))
    return home_win_prob, draw_prob, away_win_prob


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def batch_outcome_probabilities(
    matches: List[Dict[str, float]],
    max_goals: int = MAX_GOALS,
    dixon_coles_rho: float = 0.0,
    bivariate_l3: float = 0.0,
) -> List[Dict[str, float]]:
    """
    Score a list of matches and return W/D/L probabilities for each.

    Args:
        matches: List of dicts with keys 'home_pred' and 'away_pred'.
        max_goals, dixon_coles_rho, bivariate_l3: passed to poisson_matrix.

    Returns:
        List of dicts with keys 'home_win', 'draw', 'away_win'.

    Example:
        matches = [
            {"home_pred": 1.5, "away_pred": 1.1},
            {"home_pred": 2.2, "away_pred": 0.8},
        ]
        results = batch_outcome_probabilities(matches, bivariate_l3=0.12)
    """
    results = []
    for m in matches:
        mat = poisson_matrix(
            m["home_pred"], m["away_pred"],
            max_goals=max_goals,
            dixon_coles_rho=dixon_coles_rho,
            bivariate_l3=bivariate_l3,
        )
        h, d, a = match_outcome_probabilities(mat)
        results.append({"home_win": h, "draw": d, "away_win": a})
    return results