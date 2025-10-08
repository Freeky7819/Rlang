def adam_update(params, grads, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    if not state:
        state['m'] = [0.0]*len(params)
        state['v'] = [0.0]*len(params)
        state['t'] = 0
    state['t'] += 1
    for i, g in enumerate(grads):
        state['m'][i] = b1*state['m'][i] + (1-b1)*g
        state['v'][i] = b2*state['v'][i] + (1-b2)*(g*g)
        mhat = state['m'][i] / (1 - b1**state['t'])
        vhat = state['v'][i] / (1 - b2**state['t'])
        params[i] -= lr * mhat / ((vhat**0.5) + eps)
    return params, state
