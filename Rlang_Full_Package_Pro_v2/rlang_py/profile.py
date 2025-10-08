import json
def load_profile(path): return json.load(open(path,'r',encoding='utf-8'))
def load_state(path):   return json.load(open(path,'r',encoding='utf-8'))
