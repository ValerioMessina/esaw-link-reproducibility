"""Fail closed for weights, datasets, archives, keys, binaries and secrets.

This structural/content check supplements human review. It does not claim
universal anonymization. Use the external assembly privacy audit before release.
"""
import ast,hashlib,json,re,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'.py','.sh','.md','.txt','.json','.cff','.example',''}
SENSITIVE_PARTS={'private','data','datasets','models','checkpoints','outputs','results','artifacts','logs','hf_cache'}
SENSITIVE_NAMES={'dataset.json','benchmark.json','split.json','candidates.json','tasks.json',
                 'responses.json','predictions.json','runpod_state.json','process.json'}
SECRET_PATTERNS=[r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                 r'\b(?:sk|hf)_[A-Za-z0-9]{24,}\b',r'\bsk-[A-Za-z0-9_-]{24,}\b',
                 r'\bAKIA[A-Z0-9]{16}\b',r'gh[pousr]_[A-Za-z0-9]{30,}']

def check(root=ROOT):
    errors=[];count=0;size=0
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if '.git' in rel.parts:continue
        if '__pycache__' in rel.parts:continue
        if p.is_symlink():errors.append(f'Symlink: {rel}');continue
        if not p.is_file():continue
        count+=1;b=p.read_bytes();size+=len(b)
        if p.name in SENSITIVE_NAMES:errors.append(f'Forbidden artifact name: {rel}')
        if any(v in SENSITIVE_PARTS for v in rel.parts):errors.append(f'Forbidden directory: {rel}')
        if p.name not in {'.gitignore','LICENSE'} and p.suffix not in ALLOWED:errors.append(f'Forbidden extension: {rel}')
        if len(b)>300000 or b'\x00' in b:errors.append(f'Binary or oversized file: {rel}')
        try:text=b.decode('utf-8')
        except UnicodeDecodeError:errors.append(f'Non-UTF8: {rel}');continue
        if any(re.search(rx,text) for rx in SECRET_PATTERNS):errors.append(f'Possible credential: {rel}')
        if p.suffix=='.py':
            try:ast.parse(text,filename=str(rel))
            except SyntaxError:errors.append(f'Python syntax: {rel}')
    manifest=root/'provenance/source_manifest.json'
    if manifest.exists():
        for row in json.loads(manifest.read_text()):
            p=root/row['path']
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
                errors.append(f'Source hash mismatch: {row["path"]}')
    return {'passed':not errors,'files':count,'bytes':size,'errors':errors,
            'scope':'allowlisted text, syntax, source hashes, forbidden artifacts, common credential patterns'}

if __name__=='__main__':
    result=check();print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)

