def heuristic_profile(n=8, alpha=0.15, omega=0.8):
    # Slower oscillation for neural-like rhythm
    return {'chord':[0.05,0.2,0.4,0.6][:n], 'alpha':alpha, 'omega':omega}
