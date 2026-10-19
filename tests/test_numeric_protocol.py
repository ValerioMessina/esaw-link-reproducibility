"""Synthetic statistical/decoder controls; requires NumPy, no models or data."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'retriever'),str(ROOT/'scripts')]
try:import numpy as np
except ImportError:np=None

@unittest.skipIf(np is None,'NumPy not installed; run in a study environment')
class NumericProtocol(unittest.TestCase):
    def test_validation_offset_and_stable_parent_child_tie(self):
        from decode_scores import fit,decode,metrics
        # Eight fields, each with an admissible parent and a child.
        members=[np.array([2*j,2*j+1]) for j in range(8)]
        depths=np.tile([1,2],8);z=np.tile([10.,9.],(4,8));g=np.tile(np.arange(1,16,2),(4,1))
        offsets=fit(z,g,members,depths)
        self.assertEqual(offsets,[1.25]*8)  # +1 ties to parent; +1.25 first correct.
        p=decode(z,members,depths,offsets,k=1)
        self.assertEqual(metrics(p,g)['record_exact8_pct'],100)
        raw=decode(z,members,depths,[0]*8,k=1)
        self.assertEqual(metrics(raw,g)['mean_top1_pct'],0)
        tied=decode(z,members,depths,[1]*8,k=1)
        self.assertTrue(np.array_equal(tied[:,:,0],np.tile(np.arange(0,16,2),(4,1))))

    def test_bootstrap_identical_systems_zero_difference(self):
        from paired_bootstrap import compare
        hits=np.array([[1]*8,[0]*8,[1,0]*4],dtype=bool)
        result=compare(hits,hits,replicates=200)
        for k in ('mean_top1','exact8'):
            self.assertEqual(result[k]['delta_percentage_points'],0)
            self.assertEqual(result[k]['ci95'],[0,0])

    def test_bootstrap_keeps_correlated_fields_together(self):
        from paired_bootstrap import compare
        a=np.zeros((2,8),dtype=bool);b=np.array([[1]*8,[0]*8],dtype=bool)
        result=compare(a,b,replicates=1000)
        self.assertEqual(result['mean_top1']['ci95'],[0,100])
        self.assertEqual(result['exact8']['ci95'],[0,100])
        self.assertEqual(result['mean_top1']['delta_percentage_points'],50)

    def test_material_candidate_documents_are_role_conditioned(self):
        from esaw_biencoder.data import build_candidates,build_explicit_taxonomy
        tax={'tipo_luogo':{'010':'Synthetic group','011':'Synthetic child'},
             'tipo_lavoro':{'10':'Synthetic group','11':'Synthetic child'},
             'attivita_fisica_specifica':{'10':'Synthetic group','11':'Synthetic child'},
             'deviazione':{'10':'Synthetic group','11':'Synthetic child'},
             'contatto_modalita_lesione':{'10':'Synthetic group','11':'Synthetic child'},
             'agente_materiale':{'01.00':'Synthetic group','01.01':'Synthetic object'}}
        docs=build_candidates(build_explicit_taxonomy(tax),text_format='v5')
        texts=[next(c.text for c in docs[f] if c.taxonomy_code=='01.01')
               for f in ('cod_mat_atv','cod_mat_dev','cod_mat_con')]
        self.assertEqual(len(set(texts)),3)

if __name__=='__main__':unittest.main()
