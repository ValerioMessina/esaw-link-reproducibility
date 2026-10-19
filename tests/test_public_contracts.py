"""Synthetic checks: no institutional examples, no model download or API call."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'retriever'),str(ROOT/'scripts')]
import canonical_audit as audit
from check_inputs import external,inspect
from make_split import make
from run_experiment import build

class Contracts(unittest.TestCase):
    def fixture(self):
        request={'record_key':'SYNTHETIC_ONLY','fields_to_audit':[]};answer={'record_key':'SYNTHETIC_ONLY','field_audits':[]}
        for f in audit.FIELDS:
            request['fields_to_audit'].append({'field':f,'allowed_choices':[{'code':'01'}],
                'gold':{'code':'01'},'dual_top1':{'code':'01'},'gemma_top1':{'code':'01'}})
            answer['field_audits'].append({'field':f,'audited_code':'01','source':'keep_gold',
                'gold_quality':'valid_specific','confidence':'high','needs_human_review':False,'rationale':'Synthetic fixture only'})
        return request,answer

    def test_complete_fixture(self):
        p,a=self.fixture();self.assertEqual(audit.validate_audit(p,a),[])

    def test_empty_duplicate_missing_wrong_identity(self):
        p,a=self.fixture()
        for kind in ('empty','duplicate','missing','identity'):
            b=copy.deepcopy(a)
            if kind=='empty':b['field_audits']=[]
            elif kind=='duplicate':b['field_audits'].append(b['field_audits'][0])
            elif kind=='missing':b['field_audits'].pop()
            else:b['record_key']='OTHER_SYNTHETIC'
            self.assertTrue(audit.validate_audit(p,b),kind)

    def test_material_depth_not_truncated(self):
        for f in ('COD_MAT_ATV','COD_MAT_DEV','COD_MAT_CON'):
            self.assertEqual(audit.normalize_code(f,'01.02.00.00'),'01.02')
            self.assertEqual(audit.normalize_code(f,'01.02.03.00'),'')
            self.assertEqual(audit.normalize_code(f,'needs_human_review'),'')

    def test_review_flag_is_strict_boolean(self):
        p,a=self.fixture();a['field_audits'][0]['needs_human_review']=1
        self.assertTrue(audit.validate_audit(p,a))

    def test_split_disjoint_reproducible(self):
        rows=[{'id':f'SYNTHETIC-{i}'} for i in range(100)]
        a=make(rows);self.assertEqual(a,make(rows));sets=[set(v) for v in a['splits'].values()]
        self.assertEqual(sorted(map(len,sets)),[10,10,80]);self.assertEqual(len(set.union(*sets)),100)
        self.assertFalse(sets[0]&sets[1])

    def test_private_input_guard(self):
        with self.assertRaises(ValueError):external(ROOT/'private'/'dataset.json')

    def test_configs_have_only_external_generated_outputs(self):
        vals={k:'/external/'+k for k in ('dataset','taxonomy','split_file','work_dir','phase1_model','refined_model','hard_negatives','data_dir')}
        vals.update(root=str(ROOT),python=sys.executable,base_model='bert-base-uncased')
        for p in (ROOT/'configs').glob('*.json'):
            if p.name=='gemma_local_protocol.json':continue
            spec,args=build(p,vals);self.assertTrue(args);self.assertNotIn('shell',spec)
            for f in ('--output-dir','--output','--predictions-output','--summary-output'):
                if f in args:self.assertTrue(args[args.index(f)+1].startswith('/external/'))

if __name__=='__main__':unittest.main()
