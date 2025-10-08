import json, sys, math, random

def step(state, profile):
    random.seed(state["seed"])
    new_phase, new_amp = [], []
    for ph, amp, ch in zip(state["phase"], state["amp"], profile["chord"]):
        delta = profile["alpha"] * math.sin(ph * profile["omega"] + ch)
        new_phase.append(ph + delta)
        new_amp.append(amp + delta * 0.1)
    new_seed = (state["seed"] * 6364136223846793005 + 1) % (2**64)
    return {
        "state": {"phase": new_phase, "amp": new_amp, "seed": new_seed},
        "loss": sum(abs(a-b) for a,b in zip(state["phase"], new_phase))
    }

if __name__ == "__main__":
    profile = json.load(open(sys.argv[1]))
    state = json.load(open(sys.argv[2]))
    print(json.dumps(step(state, profile), indent=2))
