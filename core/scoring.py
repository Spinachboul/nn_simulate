def compute_score(value : float) -> int:
    """ Scoring between 0 and 100 based on the accuracy"""

    if value.failed:
        return 0
    score  = 0
    score += int(value.accuracy * 50)        # Max 50
    score += int(value.stability * 20)       # Max 20
    score += int(value.convergence_speed * 20)  # Max 20
    score += int((1 - value.overfitting_risk) * 10)
    return max(0, min(score, 100))