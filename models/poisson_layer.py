import numpy as np
from scipy.stats import poisson

MAX_GOALS = 7  # truncate, but we renormalize so we don't lose tail mass

def truncated_pmf(lam, max_goals):
    ks = np.arange(0, max_goals + 1)
    pmf = poisson.pmf(ks, lam)
    z = poisson.cdf(max_goals, lam)  # P(X <= max_goals)
    if z <= 0:
        # extremely tiny lam edge-case
        pmf = np.zeros_like(pmf)
        pmf[0] = 1.0
        return pmf
    return pmf / z

def poisson_matrix(home_pred, away_pred, max_goals=MAX_GOALS, dixon_coles_rho=0.0):
    home_probs = truncated_pmf(home_pred, max_goals)
    away_probs = truncated_pmf(away_pred, max_goals)

    score_matrix = np.outer(home_probs, away_probs)

    if dixon_coles_rho != 0.0:
        # Dixon–Coles tau adjustment factors for low scores
        # tau(0,0)=1 - rho*λh*λa
        # tau(0,1)=1 + rho*λh
        # tau(1,0)=1 + rho*λa
        # tau(1,1)=1 - rho
        lam_h = float(home_pred)
        lam_a = float(away_pred)

        tau = np.ones_like(score_matrix)
        tau[0, 0] = 1.0 - dixon_coles_rho * lam_h * lam_a
        tau[0, 1] = 1.0 + dixon_coles_rho * lam_h
        tau[1, 0] = 1.0 + dixon_coles_rho * lam_a
        tau[1, 1] = 1.0 - dixon_coles_rho

        # avoid negative factors if rho is too extreme
        tau = np.clip(tau, 0.0, None)

        score_matrix = score_matrix * tau

    # normalize to sum to 1
    total = score_matrix.sum()
    if total > 0:
        score_matrix = score_matrix / total

    return score_matrix

def match_outcome_probabilities(score_matrix):
    # Home win (i>j) is upper triangle if i=home goals and j=away goals
    home_win_prob = np.sum(np.tril(score_matrix, -1))
    draw_prob = np.sum(np.diag(score_matrix))
    away_win_prob = np.sum(np.triu(score_matrix, 1))

    return home_win_prob, draw_prob, away_win_prob
