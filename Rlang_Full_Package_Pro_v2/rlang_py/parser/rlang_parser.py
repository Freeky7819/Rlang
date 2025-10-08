def parse_rlang(path):
    data = {}
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'): continue
            if s.startswith('alpha'): data['alpha'] = float(s.split()[1])
            elif s.startswith('omega'): data['omega'] = float(s.split()[1])
            elif s.startswith('chord'):
                arr = s[s.find('[')+1:s.find(']')].split(',')
                data['chord'] = [float(x) for x in arr if x.strip()]
    data.setdefault('chord',[0.1,0.3,0.7]); data.setdefault('alpha',0.2); data.setdefault('omega',1.5)
    return data
