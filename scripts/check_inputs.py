"""Validate authorized external inputs without printing record contents or IDs."""
import argparse, hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TAXONOMY_SHA='f14ed4c351603d6d23e53b310f0b4d57af58737d5cd3b75016454cd0065f0366'
TAX={'cod_loc':'tipo_luogo','cod_lav':'tipo_lavoro','cod_atv':'attivita_fisica_specifica',
     'cod_mat_atv':'agente_materiale','cod_dev':'deviazione','cod_mat_dev':'agente_materiale',
     'cod_con':'contatto_modalita_lesione','cod_mat_con':'agente_materiale'}

def external(path):
    p=Path(path).expanduser().resolve()
    if p==ROOT or ROOT in p.parents:
        raise ValueError('Private inputs and generated outputs must be outside the repository')
    return p

def load_records(path):
    x=json.loads(Path(path).read_text())
    return x if isinstance(x,list) else x['results'][0]['items']

def key(row,index):
    for f in ('id','numero_caso'):
        v=row.get(f) or row.get(f.upper())
        if v is not None and str(v).strip():return str(v).strip()
    return f'index:{index}'

def inspect(dataset,taxonomy,split_file=None,strict=True):
    rows=load_records(dataset);tax=json.loads(Path(taxonomy).read_text())
    digest=hashlib.sha256(Path(taxonomy).read_bytes()).hexdigest()
    if strict and digest!=TAXONOMY_SHA:raise ValueError('Taxonomy differs from frozen study version')
    ids=[key(r,i) for i,r in enumerate(rows)]
    if len(ids)!=len(set(ids)):raise ValueError('Dataset identities are not unique')
    invalid=0
    for row in rows:
        for f,t in TAX.items():
            raw=str(row.get(f) or row.get(f.upper()) or '').strip()
            if t=='agente_materiale':
                if not re.fullmatch(r'\d{2}\.\d{2}(?:\.00\.00)?',raw):invalid+=1;continue
                raw=raw[:5]
            if raw not in tax[t]:invalid+=1
    if invalid:raise ValueError(f'Invalid or unsupported target assignments: {invalid}')
    sizes=None
    if split_file:
        payload=json.loads(Path(split_file).read_text());splits=payload.get('splits',payload)
        members={name:list(map(str,splits[name])) for name in ('train','validation','test')}
        if any(len(v)!=len(set(v)) for v in members.values()):raise ValueError('Duplicate split IDs')
        parts=[set(v) for v in members.values()]
        if any(parts[i]&parts[j] for i in range(3) for j in range(i)):raise ValueError('Overlapping split IDs')
        if set.union(*parts)!=set(ids):raise ValueError('Split does not exactly cover dataset')
        sizes={k:len(v) for k,v in members.items()}
        if strict and (len(rows)!=50000 or sizes!={'train':40000,'validation':5000,'test':5000}):
            raise ValueError('Unexpected historical study split sizes')
    return {'records':len(rows),'split_sizes':sizes,'taxonomy_sha256':digest,
            'dataset_sha256':hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
            'strict_study_inputs':strict,'valid':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--taxonomy',required=True)
    p.add_argument('--split-file');p.add_argument('--allow-new-study-inputs',action='store_true')
    a=p.parse_args();print(json.dumps(inspect(external(a.dataset),external(a.taxonomy),
         external(a.split_file) if a.split_file else None,not a.allow_new_study_inputs),indent=2))
if __name__=='__main__':main()

