from pathlib import Path
import importlib.util,json,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('ref',ROOT/'htc_clip_local_benchmark_20261017/run_local.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
rows=m.read_jsonl(m.D/'esaw_test.json');gold=m.gold(rows);manifest=json.loads((R/'frozen_checkpoints_decoders.json').read_text());out={};table=[]
for seed in [3,13,21,42,87]:
 d=R/f'seed{seed}';v=np.load(d/'test_scores.npz');assert list(v['record_keys'])==[x['record_key'] for x in rows];assert np.array_equal(v['gold_ids'],gold);metric=json.loads((d/'metrics.json').read_text());preds=[json.loads(x) for x in (d/'predictions.jsonl').open()];assert len(preds)==5000
 for mode,key in [('raw','A_raw'),('calibrated','B_validation_calibrated')]:
  hits=np.zeros((5000,8,3),bool)
  expected=m.decode(v['logits'],None if mode=='raw' else manifest[str(seed)]['offsets'])
  assert np.array_equal(expected,np.array([p[mode+'_top10'] for p in preds]))
  for i,p in enumerate(preds):
   assert p['record_key']==rows[i]['record_key'] and p['gold']==gold[i].tolist()
   for j in range(8):
    rank=p[mode+'_top10'][j];assert len(set(rank))==10;assert set(rank)<=set(m.H.field_leaves[m.FIELDS[j]])
    for n,k in enumerate([1,5,10]):hits[i,j,n]=int(gold[i,j]) in rank[:k]
  for n,k in enumerate([1,5,10]):
   assert abs(hits[:,:,n].mean()*100-metric[key][f'mean_top{k}_pct'])<1e-9
   assert abs(hits[:,:,n].all(1).mean()*100-metric[key][f'record_all8_top{k}_pct'])<1e-9
 out[str(seed)]={'passed':True,'records':5000,'scores_sha256':m.sha(d/'test_scores.npz'),'predictions_sha256':m.sha(d/'predictions.jsonl')}
 for key,title in [('A_raw','Raw'),('B_validation_calibrated','Calibrato')]:
  a=metric[key];table.append(f"| HGCLR | {seed} | {'completo' if metric['training_status']['complete'] else 'incompleto'} | {metric['best']['epoch']} | {title} | {a['mean_top1_pct']:.4f} | {a['mean_top5_pct']:.4f} | {a['mean_top10_pct']:.4f} | {a['record_exact8_pct']:.4f} |")
# Existing tests: reuse only after matching checkpoint hashes and independently checking arithmetic.
existing={}
for name,d in [('HTC-CLIP',ROOT/'htc_clip_local_benchmark_20261017'),('HiSR',ROOT/'paper_closure_20261018/checkpoint_decoder/hisr'),('HyILR',ROOT/'paper_closure_20261018/checkpoint_decoder/hyilr')]:
 p=json.loads((d/'protocol.json').read_text());ck=Path(p['checkpoint']);assert m.sha(ck)==p['checkpoint_sha256'];v=np.load(d/'test_scores.npz');assert list(v['record_keys'])==[x['record_key'] for x in rows];assert np.array_equal(v['gold_ids'],gold);a=json.loads((d/'test_metrics.json').read_text());o=json.loads((d/'decoder_frozen.json').read_text())['offsets']
 for key,offset in [('A_raw',None),('B_validation_calibrated',o)]:
  calc=m.metrics(m.decode(v['logits'],offset),gold)
  for k in ['mean_top1_pct','mean_top5_pct','mean_top10_pct','record_exact8_pct']:assert abs(calc[k]-a[key][k])<1e-9
 existing[name]={'verified':True,'checkpoint':str(ck),'checkpoint_sha256':p['checkpoint_sha256'],'metrics':a,'source':str(d)}
 for key,title in [('A_raw','Raw'),('B_validation_calibrated','Calibrato')]:
  x=a[key];table.append(f"| {name} | {3 if name=='HTC-CLIP' else 13} | storico | {50 if name=='HTC-CLIP' else 15 if name=='HiSR' else 6} | {title} | {x['mean_top1_pct']:.4f} | {x['mean_top5_pct']:.4f} | {x['mean_top10_pct']:.4f} | {x['record_exact8_pct']:.4f} |")
save(R/'verification.json',{'passed':True,'new_hgclr_tests':out,'existing_tests':existing})
text='''# Test delle baseline gerarchiche — 21 ottobre 2026

Valutati i cinque checkpoint best-Macro-F1 disponibili del nuovo HGCLR, selezionati sulla validation. Quattro training completi; seed87 incompleto per budget. Tutti valutati sui medesimi 5.000 record test / 40.000 decisioni. Nessuna selezione dei seed sul test. Decoder calibrato con offset fissati sulla validation prima di leggere il test; analisi retrospettiva.

Il checkpoint selezionato sulla validation viene usato al posto dell’ultima epoca, coerentemente con il protocollo originale. Non è stato effettuato nuovo training. L’inferenza usa encoder e classifier originali con caricamento strict e verifica numerica sui logits validation.

HiSR, HyILR e HTC-CLIP avevano già test completi: hash checkpoint, identità dei record e metriche ricalcolate verificati; risultati riutilizzati. Le configurazioni di tuning interrotte non sostituiscono i checkpoint finali.

Tutti i valori sono percentuali. Exact8 richiede otto campi corretti contemporaneamente.

| Metodo | Seed | Training | Epoca checkpoint | Decoder | Top1 | Top5 | Top10 | Exact8 |
|---|---:|---|---:|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'\n\nIl protocollo confermativo HGCLR a cinque training completi resta incompleto: questi risultati documentano i checkpoint disponibili e non vanno presentati come quel benchmark completato. Il manoscritto non è stato modificato.\n'
(R/'REPORT.md').write_text(text)
target=Path('/Volumes/university data/tesi/benchs/reports/baseline_test_20261021');shutil.copytree(R,target,dirs_exist_ok=True)
for p in R.rglob('*'):
 if p.is_file():assert m.sha(p)==m.sha(target/p.relative_to(R))
save(R/'ssd_backup_verified.json',{'path':str(target),'all_files_sha256_equal':True});shutil.copy2(R/'ssd_backup_verified.json',target/'ssd_backup_verified.json');print(text)
