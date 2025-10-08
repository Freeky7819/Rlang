# tools/phase_view.py
# Read PhaseHistory binary and plot + detect lock.
# Format:
#   uint64 K, uint32[K] indices, uint64 T, double[T] times, then K * double[T] samples.

import struct, sys, math
import numpy as np
import matplotlib.pyplot as plt

def read(path):
    with open(path,"rb") as f:
        K = struct.unpack("<Q", f.read(8))[0]
        idx = list(struct.unpack("<" + "I"*K, f.read(4*K)))
        T = struct.unpack("<Q", f.read(8))[0]
        times = np.frombuffer(f.read(8*T), dtype="<f8").copy()
        series = []
        for k in range(K):
            series.append(np.frombuffer(f.read(8*T), dtype="<f8").copy())
    return idx, times, series

def wrap_err(p,q,a,b):
    e = (p*a - q*b + math.pi) % (2*math.pi) - math.pi
    return e

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python phase_view.py phase_trace.bin [p:q pairs like 5:4,6:5,3:2]")
        sys.exit(1)
    idx, t, ser = read(sys.argv[1])
    print("Indices:", idx, "frames:", len(t))

    # Default: check consecutive pairs 1:1
    pairs = []
    if len(sys.argv)>=3:
        for tok in sys.argv[2].split(","):
            pq = tok.strip().split(":")
            pairs.append((int(pq[0]), int(pq[1])))
    else:
        for k in range(len(idx)-1):
            pairs.append((1,1))

    # Build errors
    errs = []
    for i,(p,q) in enumerate(pairs):
        a = ser[i]; b = ser[(i+1)%len(ser)] if len(pairs)>1 else ser[i+1]
        e = np.array([wrap_err(p,q,aa,bb) for aa,bb in zip(a,b)])
        errs.append(e)

    # Plot
    plt.figure(figsize=(12,6))
    for i,e in enumerate(errs):
        plt.plot(t, e, label=f"Pair {i} err")
    for thr in [0.03,-0.03]:
        plt.axhline(thr, linestyle="--", alpha=0.5)
    plt.title("Phase errors")
    plt.xlabel("time (s)"); plt.ylabel("rad")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/data/phase_errors.png", dpi=140)
    print("Saved /mnt/data/phase_errors.png")
