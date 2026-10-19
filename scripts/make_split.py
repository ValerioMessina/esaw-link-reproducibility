"""Original seed-13 split generation. Output is private and must stay external."""
import argparse,json,random
from check_inputs import external,load_records,key

def make(rows,seed=13):
    keys=[key(r,i) for i,r in enumerate(rows)]
    if len(keys)!=len(set(keys)):raise ValueError('Nonunique input identities')
    random.Random(seed).shuffle(keys);n=max(1,int(len(keys)*.1))
    if len(keys)<10:raise ValueError('At least ten records required')
    return {'seed':seed,'validation_ratio':.1,'test_ratio':.1,
        'split_key':'id, fallback numero_caso, fallback index',
        'splits':{'test':keys[:n],'validation':keys[n:2*n],'train':keys[2*n:]}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=13);a=p.parse_args();out=external(a.output)
    payload=make(load_records(external(a.dataset)),a.seed);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(payload,f,indent=2)
    print(json.dumps({'sizes':{k:len(v) for k,v in payload['splits'].items()},'written':True}))
if __name__=='__main__':main()

