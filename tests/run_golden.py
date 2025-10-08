import json, subprocess

def close(a, b, atol=1e-9, rtol=1e-9):
    return abs(a-b) <= (atol + rtol*abs(b))

def compare(res1, res2):
    p1, p2 = res1["state"]["phase"], res2["state"]["phase"]
    a1, a2 = res1["state"]["amp"], res2["state"]["amp"]
    for i, (x,y) in enumerate(zip(p1,p2)):
        if not close(x,y): return False
    for i, (x,y) in enumerate(zip(a1,a2)):
        if not close(x,y): return False
    return res1["state"]["seed"] == res2["state"]["seed"]

if __name__ == "__main__":
    py = json.loads(subprocess.check_output(
        ["python", "rlang_py/rlang_ref.py", "tests/golden/profile.json", "tests/golden/state.json"]
    ))
    cpp = json.loads(subprocess.check_output(
        ["./build/rlang_cli", "tests/golden/profile.json", "tests/golden/state.json"]
    ))
    print("OK" if compare(py, cpp) else "MISMATCH")
