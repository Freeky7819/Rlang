import json, subprocess, sys, os, re

base = os.path.dirname(__file__) + "/.."
py = subprocess.check_output([sys.executable, "-m", "rlang_py.cli", "run-step",
    "--profile", "tests/golden/profile.json", "--state", "tests/golden/state.json"], cwd=base).decode()

# Build native
if not os.path.exists(os.path.join(base, "build")):
    subprocess.check_call(["cmake", "-S", ".", "-B", "build", "-DRLANG_ENABLE_AVX2=ON"], cwd=base)
    subprocess.check_call(["cmake", "--build", "build", "-j"], cwd=base)

cpp = subprocess.check_output(["./build/rlang_cli", "tests/golden/profile.txt", "tests/golden/state.txt"], cwd=base).decode()

phase = list(map(float, re.search(r'"phase":\s*\[([^\]]+)\]', cpp).group(1).split(',')))
amp   = list(map(float, re.search(r'"amp":\s*\[([^\]]+)\]', cpp).group(1).split(',')))
seed  = int(re.search(r'"seed":\s*(\d+)', cpp).group(1))

j = json.loads(py)
ok = all(abs(a-b)<1e-8 for a,b in zip(j['state']['phase'], phase)) and      all(abs(a-b)<1e-8 for a,b in zip(j['state']['amp'], amp)) and      (int(j['state']['seed'])==seed)

print("OK" if ok else "MISMATCH")
