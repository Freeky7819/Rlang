def heuristic_profile(n=8, alpha=0.12, omega=0.9):
    return {'chord':[0.1,0.25,0.5,0.75][:n], 'alpha':alpha, 'omega':omega}
