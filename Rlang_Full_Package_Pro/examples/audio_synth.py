import argparse, wave, struct, math, json
from rlang_py.interpreter import step

def render_wav(profile, seconds=4, samplerate=44100, out='out.wav'):
    state = {'phase':[0.0]*len(profile['chord']), 'amp':[0.0]*len(profile['chord']), 'seed':1}
    buf = []
    dt = 1.0/samplerate
    for n in range(int(seconds*samplerate)):
        # evolve once per sample (toy)
        st = step(state, profile)['state']
        state = st
        # mix as simple sum of sines using phases as running angles
        s = 0.0
        for i,ph in enumerate(state['phase']):
            s += math.sin(ph)
        s /= len(state['phase']) or 1
        # scale to int16
        buf.append(int(max(-1.0, min(1.0, s)) * 32767))

    with wave.open(out, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(samplerate)
        w.writeframes(b''.join(struct.pack('<h', x) for x in buf))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='examples/profile.json')
    ap.add_argument('--seconds', type=int, default=4)
    ap.add_argument('--samplerate', type=int, default=44100)
    ap.add_argument('--out', default='out.wav')
    a = ap.parse_args()
    prof = json.load(open(a.profile))
    render_wav(prof, a.seconds, a.samplerate, a.out)
    print('Wrote', a.out)
