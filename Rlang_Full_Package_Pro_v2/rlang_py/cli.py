import argparse, json
from .interpreter import step
from .profile import load_profile, load_state
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run-step'])
    ap.add_argument('--profile', required=True)
    ap.add_argument('--state', required=True)
    a = ap.parse_args()
    if a.cmd=='run-step':
        print(json.dumps(step(load_state(a.state), load_profile(a.profile)), indent=2))
if __name__=='__main__': main()
