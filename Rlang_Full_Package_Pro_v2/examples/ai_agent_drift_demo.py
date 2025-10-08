import numpy as np, matplotlib.pyplot as plt
def simulate_baseline(u, g, d, T, rng):
    eta, gamma, noise = 0.08, 0.06, 0.01
    v1 = rng.normal(size=d); v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=d); v2 -= v1*np.dot(v1, v2); v2 /= np.linalg.norm(v2)
    theta = 0.02
    def rotate(x):
        a, b = np.dot(x, v1), np.dot(x, v2)
        ar = a*np.cos(theta) - b*np.sin(theta)
        br = a*np.sin(theta) + b*np.cos(theta)
        return x + (ar - a)*v1 + (br - b)*v2
    b = u.copy(); bh=[b.copy()]
    for _ in range(T):
        step = eta*(g - b); crit = gamma*(rotate(b) - b)
        eps = noise * rng.normal(size=d); b = b + step + crit + eps
        b /= np.linalg.norm(b); bh.append(b.copy())
    return {"b": bh}
def simulate_rlang(u, g, d, T, rng):
    eta, gamma, noise = 0.08, 0.06, 0.01
    v1 = rng.normal(size=d); v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=d); v2 -= v1*np.dot(v1, v2); v2 /= np.linalg.norm(v2)
    theta = 0.02
    def rotate(x):
        a, b = np.dot(x, v1), np.dot(x, v2)
        ar = a*np.cos(theta) - b*np.sin(theta)
        br = a*np.sin(theta) + b*np.cos(theta)
        return x + (ar - a)*v1 + (br - b)*v2
    alpha, omega, phi = 0.12, 0.9, 0.4
    b = u.copy(); bh=[b.copy()]
    for t in range(1, T+1):
        step = eta*(g - b); crit = gamma*(rotate(b) - b)
        eps = noise * rng.normal(size=d)
        u_proj = u * (np.dot(b, u) / (np.linalg.norm(u)**2))
        res = -alpha * np.sin(omega*np.log(t+1.0) + phi) * (b - u_proj)
        b = b + step + crit + eps + res; b /= np.linalg.norm(b); bh.append(b.copy())
    return {"b": bh}
def cos(a,b): return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))
if __name__=='__main__':
    d, T = 64, 100; rng = np.random.default_rng(42)
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    g = rng.normal(size=d); g /= np.linalg.norm(g)
    base = simulate_baseline(u,g,d,T,rng); rl = simulate_rlang(u,g,d,T,np.random.default_rng(42))
    drift_b = [1 - cos(x,u) for x in base['b']]; drift_r = [1 - cos(x,u) for x in rl['b']]
    align_b = [cos(x,g) for x in base['b']]; align_r = [cos(x,g) for x in rl['b']]
    print('Final:', {'baseline':{'drift':drift_b[-1],'align':align_b[-1]},
                     'rlang':{'drift':drift_r[-1],'align':align_r[-1]}})
    import matplotlib.pyplot as plt
    plt.figure(); plt.plot(drift_b,label='Baseline drift'); plt.plot(drift_r,label='Rlang drift')
    plt.xlabel('iter'); plt.ylabel('1 - cos(b,u)'); plt.legend(); plt.tight_layout(); plt.savefig('ai_drift.png', dpi=140)
    plt.figure(); plt.plot(align_b,label='Baseline align'); plt.plot(align_r,label='Rlang align')
    plt.xlabel('iter'); plt.ylabel('cos(b,g)'); plt.legend(); plt.tight_layout(); plt.savefig('ai_align.png', dpi=140)
