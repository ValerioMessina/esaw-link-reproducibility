from pathlib import Path
from collections import defaultdict
import json,hashlib,re,struct,mmap,math,sys
O=Path(__file__).resolve().parent;R=Path('/Volumes/university data/tesi/benchs/dual encoder');sys.path.insert(0,str(R))
from esaw_biencoder.data import build_explicit_taxonomy,build_candidates,parent_code_for,material_taxonomy_code,material_output_code
p=R/'tassonomia.json';before=p.read_bytes();sha=hashlib.sha256(before).hexdigest();assert sha=='f14ed4c351603d6d23e53b310f0b4d57af58737d5cd3b75016454cd0065f0366'
duplicates=[]
def pairs_hook(pairs):
 d={}
 for k,v in pairs:
  if k in d:duplicates.append(k)
  d[k]=v
 return d
tax=json.loads(before,object_pairs_hook=pairs_hook);assert not duplicates
issues=[];label_dups=[];parents={};invalid=[];orphans=[]
for n,codes in tax.items():
 groups=defaultdict(list)
 for c,label in codes.items():
  pattern=r'\d{2}\.\d{2}' if n=='agente_materiale' else (r'\d{3}' if n=='tipo_luogo' else r'\d{2}')
  if not re.fullmatch(pattern,c) or not isinstance(label,str) or not label.strip():invalid.append([n,c])
  groups[' '.join(label.casefold().split())].append(c)
  parent=parent_code_for(n,c,codes);parents[n+':'+c]=parent
  seen={c};cur=parent
  while cur:
   assert cur in codes and cur not in seen,(n,c,cur)
   seen.add(cur);cur=parent_code_for(n,cur,codes)
  if parent is None and n!='agente_materiale' and not c.endswith('0') and c not in {'99','099','999'}:orphans.append([n,c])
  if n=='agente_materiale':assert material_taxonomy_code(material_output_code(c))==c
  m=re.search(r'gruppo\s+(\d+)',label,re.I)
  if m and n!='agente_materiale' and int(m.group(1))!=int(c[:-1]+'0'):issues.append({'taxonomy':n,'code':c,'label':label,'mentioned_group':m.group(1),'code_expected_group':c[:-1]+'0'})
 for lab,cs in groups.items():
  if len(cs)>1:label_dups.append({'taxonomy':n,'codes':cs,'label':lab})
assert not invalid and not orphans,(invalid,orphans)
assert len(issues)==1 and issues[0]['code']=='19'
corrected=json.loads(before);old=corrected['attivita_fisica_specifica']['19'];new=old.replace('gruppo 20','gruppo 10');assert new!=old
corrected['attivita_fisica_specifica']['19']=new
(O/'tassonomia.corrected-activity19.v1.json').write_text(json.dumps(corrected,indent=2,ensure_ascii=False)+'\n')
patch=[{'op':'test','path':'/attivita_fisica_specifica/19','value':old},{'op':'replace','path':'/attivita_fisica_specifica/19','value':new}]
(O/'activity19.patch.json').write_text(json.dumps(patch,indent=2,ensure_ascii=False)+'\n')
cs=build_candidates(build_explicit_taxonomy(corrected),text_format='v5')
originalcs=build_candidates(build_explicit_taxonomy(tax),text_format='v5')
changes=[]
for k,vals in originalcs.items():
 assert len(vals)==len(cs[k]);assert [x.code for x in vals]==[x.code for x in cs[k]]
 for a,b in zip(vals,cs[k]):
  assert a.parent_code==b.parent_code
  if a.text!=b.text:changes.append({'field':k,'code':a.code,'old_text':a.text,'new_text':b.text})
assert len(changes)==1 and changes[0]['code']=='19'
# Inspect and hash tensors without allocating/loading models into GPU or modifying weights.
models=['esaw-biencoder-v5-qwen3-0.6b-mnrl-seq256','esaw-biencoder-v6-qwen3-0.6b-mnrl-hardneg','esaw-biencoder-v6-ablation-hardneg8']
summaries=[];tensor_hashes={}
for name in models:
 d=R/'models'/name;c=json.loads((d/'training_config.json').read_text());assert c['trainable_last_n_layers']==0
 w=d/'model.safetensors'
 with w.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));start=8+n
  with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
   view=memoryview(mm);hashes={}
   for k,item in header.items():
    if k=='__metadata__':continue
    a,b=item['data_offsets'];sliceview=view[start+a:start+b];hashes[k]=hashlib.sha256(sliceview).hexdigest();sliceview.release()
   view.release()
 keys=set(hashes);adapter=[k for k in keys if 'lora' in k.lower() or 'adapter' in k.lower()];assert not adapter
 layers=sorted({int(re.search(r'layers\.(\d+)\.',k).group(1)) for k in keys if re.search(r'layers\.(\d+)\.',k)})
 info={'model':name,'trainable_last_n_layers':c['trainable_last_n_layers'],'epochs':c['epochs'],'learning_rate':c['learning_rate'],'optimizer':c['optimizer'],'dtype':c['torch_dtype'],'stored_tensor_count':len(keys),'stored_parameter_elements':sum(math.prod(header[k]['shape']) for k in keys),'transformer_layers':layers,'adapter_parameter_keys':adapter,'adapter_config_present':(d/'adapter_config.json').exists(),'modules':json.loads((d/'modules.json').read_text())}
 if tensor_hashes:
  base=tensor_hashes[models[0]];assert keys==set(base)
  changed=[k for k in keys if hashes[k]!=base[k]];changed_layers=sorted({int(re.search(r'layers\.(\d+)\.',k).group(1)) for k in changed if re.search(r'layers\.(\d+)\.',k)})
  info.update(changed_tensors_vs_phase1=len(changed),layers_with_changed_weights_vs_phase1=changed_layers,embedding_changed_vs_phase1=hashes.get('embed_tokens.weight')!=base.get('embed_tokens.weight'))
 tensor_hashes[name]=hashes;summaries.append(info)
assert p.read_bytes()==before
result={'taxonomy_original_sha256':sha,'taxonomy_original_unchanged':True,'duplicate_json_keys':duplicates,'invalid_code_formats_or_empty_labels':invalid,'unexpected_nonroot_orphans':orphans,'cycles':[],'inventory_counts':{k:len(v) for k,v in tax.items()},'original_documents':sum(len(v) for v in originalcs.values()),'normalized_label_duplicates':label_dups,'label_group_mismatches':issues,'correction_status':'prepared separately; NOT used for any reported training, benchmark or active runner','corrected_taxonomy_sha256':hashlib.sha256((O/'tassonomia.corrected-activity19.v1.json').read_bytes()).hexdigest(),'candidate_text_changes_under_corrected_version':changes,'fine_tuning_models':summaries,'scope':'structural audit and one official label correction; not an exhaustive semantic certification of every description','official_reference':{'url':'https://ec.europa.eu/eurostat/documents/3859598/5926181/KS-RA-12-102-EN.PDF/56cd35ba-1e8a-4af3-9f9a-b3c47611ff1c','printed_page':26,'pdf_page_zero_based':25,'code19_group':'10','code29_group':'20'}}
(O/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');(O/'tensor_hashes.json').write_text(json.dumps(tensor_hashes,indent=2)+'\n')
print(json.dumps({'structural_status':'PASS','semantic_issue':issues,'corrected_candidate_changes':len(changes),'original_unchanged':True,'fine_tuning':[{'model':x['model'],'stored_parameters':x['stored_parameter_elements'],'layers':len(x['transformer_layers']),'changed_layers_vs_phase1':len(x.get('layers_with_changed_weights_vs_phase1',[])),'adapter_keys':len(x['adapter_parameter_keys'])} for x in summaries]},indent=2))
