import math, json

def step(state, profile):
    new_phase, new_amp = [], []
    for ph, am, ch in zip(state['phase'], state['amp'], profile['chord']):
        delta = profile['alpha'] * math.sin(ph * profile['omega'] + ch)
        new_phase.append(ph + delta)
        new_amp.append(am + 0.1 * delta)
    seed = (state['seed'] * 6364136223846793005 + 1) % (2**64)
    return {'state': {'phase': new_phase, 'amp': new_amp, 'seed': seed}}
