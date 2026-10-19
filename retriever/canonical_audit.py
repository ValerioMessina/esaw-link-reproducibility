"""Single offline audit validator/loader. Never calls providers.

Strict live validation requires the original schema. Archival loading permits only
explicit legacy representations; missing metadata stays missing and is counted.
Abstentions and historically invalid rows never become evaluation labels.
"""
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

FIELDS = ('COD_LOC','COD_LAV','COD_ATV','COD_MAT_ATV','COD_DEV','COD_MAT_DEV','COD_CON','COD_MAT_CON')
TAX = dict(zip(FIELDS,('tipo_luogo','tipo_lavoro','attivita_fisica_specifica','agente_materiale','deviazione','agente_materiale','contatto_modalita_lesione','agente_materiale')))
ALIASES = ('field_audits','audited_fields','audits','fields_to_audit')
SOURCES = {'keep_gold','dual','gemma','other_top10','needs_human_review'}
QUALITIES = {'valid_specific','too_general_parent','no_information_but_inferable','wrong_variable','ambiguous'}
REQUIRED = {'field','audited_code','source','gold_quality','confidence','needs_human_review','rationale'}

def normalize_code(field, code):
    if not isinstance(code,str): return ''
    if TAX.get(field) == 'agente_materiale':
        return code[:5] if re.fullmatch(r'\d{2}\.\d{2}(?:\.00\.00)?',code) else ''
    return code

def extract_items(audit, legacy=False):
    if not isinstance(audit,Mapping): return None, ['audit_not_object'], None
    names = [n for n in ALIASES if n in audit]
    direct = [f for f in FIELDS if f in audit]
    if len(names) + bool(direct) != 1: return None,['ambiguous_or_missing_representation'],None
    if direct:
        if not legacy: return None,['noncanonical_representation'],None
        items=[]
        for f in direct:
            x=audit[f]
            if not isinstance(x,Mapping) or ('field' in x and x['field'] != f):return None,['malformed_direct_fields'],None
            items.append({'field':f,**x})
        return items,[], 'dict_fields'
    n=names[0]
    if n != 'field_audits' and not legacy:return None,['noncanonical_representation'],None
    if not isinstance(audit[n],list): return None,['field_audits_not_array'],n
    return audit[n],[],n

def validate_items(payload, audit, legacy=False, taxonomy=None):
    errors=[]; missing=Counter()
    if not isinstance(payload,Mapping):return ['payload_not_object'],None,missing,None
    if not legacy and (not isinstance(payload.get('record_key'),str) or not payload['record_key'].strip()):errors.append('missing_or_invalid_payload_record_key')
    items,err,rep=extract_items(audit,legacy);errors.extend(err)
    if items is None:return errors,None,missing,rep
    if not legacy and set(audit)-{'record_key','field_audits'}:errors.append('unexpected_audit_properties')
    requested=payload.get('fields_to_audit')
    if not isinstance(requested,list) or not requested:return ['missing_requested_fields'],None,missing,rep
    if any(not isinstance(x,Mapping) or not isinstance(x.get('field'),str) or x.get('field') not in FIELDS for x in requested):return ['invalid_requested_fields'],None,missing,rep
    req=[x['field'] for x in requested]
    if len(set(req))!=len(req):errors.append('duplicate_requested_fields')
    pmap={x['field']:x for x in requested}
    if not legacy or 'record_key' in audit:
        if not isinstance(audit.get('record_key'),str) or audit.get('record_key') != str(payload.get('record_key','')):errors.append('record_key_mismatch')
    else:missing['record_key']+=1
    found=[];fmap={}
    for x in items:
        if not isinstance(x,Mapping):errors.append('item_not_object');continue
        f=x.get('field');found.append(f)
        if not isinstance(f,str) or f not in pmap:errors.append('unexpected_field');continue
        fmap[f]=dict(x)
        absent=REQUIRED-set(x)
        missing.update(absent)
        if absent and not legacy:errors.append('missing_required_properties')
        if not legacy and set(x)-REQUIRED:errors.append('unexpected_properties')
        code=x.get('audited_code');src=x.get('source')
        if not isinstance(code,str) or not code:errors.append('missing_or_invalid_code')
        if not isinstance(src,str) or src not in SOURCES:errors.append('invalid_source')
        if 'gold_quality' in x and (not isinstance(x['gold_quality'],str) or x['gold_quality'] not in QUALITIES):errors.append('invalid_quality')
        if 'confidence' in x and (not isinstance(x['confidence'],str) or x['confidence'] not in {'high','medium','low'}):errors.append('invalid_confidence')
        if 'needs_human_review' in x and type(x['needs_human_review']) is not bool:errors.append('invalid_review_flag')
        if 'rationale' in x and not isinstance(x['rationale'],str):errors.append('invalid_rationale')
        p=pmap[f];choices=p.get('allowed_choices')
        if not isinstance(choices,list) or not choices:errors.append('missing_allowed_choices');choices=[]
        allowed={c.get('code') for c in choices if isinstance(c,Mapping) and isinstance(c.get('code'),str)}
        if src=='needs_human_review':
            if x.get('needs_human_review') is not True:errors.append('inconsistent_abstention_flag')
            # Abstention is schema-valid only with an allowed provisional code;
            # loader below independently excludes it from canonical labels.
        if not isinstance(code,str) or code not in allowed:errors.append('outside_allowed_choices')
        expected={'keep_gold':'gold','dual':'dual_top1','gemma':'gemma_top1'}.get(src) if isinstance(src,str) else None
        if expected and code != p.get(expected,{}).get('code'):errors.append('source_code_mismatch')
        if taxonomy is not None and (not normalize_code(f,code) or normalize_code(f,code) not in taxonomy[TAX[f]]):errors.append('invalid_taxonomy_code')
    if len(set(str(f) for f in found)) != len(found):errors.append('duplicate_fields')
    if set(str(f) for f in found)!=set(req):errors.append('incomplete_or_extra_fields')
    return sorted(set(errors)),fmap,missing,rep

def validate_audit(payload,audit):
    return validate_items(payload,audit)[0]

def load_canonical(path,taxonomy):
    with Path(path).open() as handle:
        rows=[json.loads(l) for l in handle if l.strip()]
    keys=Counter(str(r.get('record_key','')) for r in rows)
    accepted={};stats=Counter();metadata_missing=Counter();representation=Counter();rejected=[]
    for line,r in enumerate(rows,1):
        key=str(r.get('record_key',''));errors=[]
        if not key:errors.append('missing_record_key')
        if keys[key]>1:errors.append('duplicate_record_key')
        if r.get('validation_errors'):errors.append('historical_validation_errors')
        p=r.get('request',{})
        if str(p.get('record_key','')) != key:errors.append('row_request_key_mismatch')
        err,items,missing,rep=validate_items(p,r.get('audit'),legacy=True,taxonomy=taxonomy)
        errors.extend(err)
        representation[rep or 'unknown']+=1
        if items is not None:
            if set(items)!=set(FIELDS):errors.append('evaluation_requires_eight_fields')
            if any(x.get('source')=='needs_human_review' for x in items.values()):errors.append('abstention')
        if errors:
            for e in set(errors):stats[e]+=1
            rejected.append({'line':line,'record_key':key,'reasons':sorted(set(errors))})
            continue
        metadata_missing.update(missing)
        accepted[key]={'labels':{f:normalize_code(f,items[f]['audited_code']) for f in FIELDS},'items':items,'missing_metadata':dict(missing),'representation':rep,'review_pending':any(x.get('needs_human_review') is True for x in items.values())}
    return accepted,{'rows':len(rows),'accepted_records':len(accepted),'rejection_counts_nonexclusive':dict(stats),'representations_all_rows':dict(representation),'missing_metadata_accepted':dict(metadata_missing)},rejected

def load_audited_gold(path,taxonomy):
    return {k:v['labels'] for k,v in load_canonical(path,taxonomy)[0].items()}
