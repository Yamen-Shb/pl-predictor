import numpy as np
from scipy.stats import poisson

MAX_GOALS = 7 # very rare scoreline, but not impossible

def poisson_matrix(home_pred, away_pred, max_goals=MAX_GOALS):
    # Returns a matrix of probabilities for each possible scoreline
    home_probs = poisson.pmf(np.arange(0, max_goals+1), home_pred)
    away_probs = poisson.pmf(np.arange(0, max_goals+1), away_pred)
    
    score_matrix = np.outer(home_probs, away_probs)
    return score_matrix

def match_outcome_probabilities(score_matrix):
    # Returns probabilities for Home win, Draw, Away win
    home_win_prob = np.sum(np.tril(score_matrix, -1))  # Lower triangle: home > away
    draw_prob = np.sum(np.diag(score_matrix))
    away_win_prob = np.sum(np.triu(score_matrix, 1))  # Upper triangle: away > home
    return home_win_prob, draw_prob, away_win_prob
