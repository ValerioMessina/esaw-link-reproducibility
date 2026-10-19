"""Test stochastic training parity, including the previously disabled BERT dropout."""
from pathlib import Path
import sys,copy,tempfile,json
from unittest.mock import patch
import torch
from torch import nn
from transformers import BertConfig
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'vendor'))
from model.contrast import ContrastModel
from esaw_htc.hgclr_model import Hgclr
class Tokenizer:
 pad_token_id=0
 def decode(self,ids):return str(ids[0])
 def __call__(self,names,**kw):return {'input_ids':[[1,int(v)+2,2] for v in names]}
def clone(source):
 m=Hgclr.__new__(Hgclr);nn.Module.__init__(m)
 m.encoder=copy.deepcopy(source.bert);m.dropout=copy.deepcopy(source.dropout);m.classifier=copy.deepcopy(source.classifier);m.graph_encoder=copy.deepcopy(source.graph_encoder);m.projector=copy.deepcopy(source.contrastive_lossfct.transform);m.contrastive_weight=.05;m.temperature=1.;return m

def corrected_positive(self,input_ids,labels,attention_mask):
 # Reuse the original view's raw word embeddings without drawing an unused
 # embedding dropout mask. Exactly the tensor consumed by the author graph.
 raw=self.encoder.embeddings.word_embeddings(input_ids)
 return self.graph_encoder(raw,attention_mask,labels,lambda ids:self.encoder.embeddings(input_ids=ids)[0])

def legacy_positive(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
 _, word_embeddings = self.encoder.embeddings(input_ids=input_ids, token_type_ids=torch.zeros_like(input_ids))

 def embed_label_ids(label_ids: torch.Tensor) -> torch.Tensor:
     return self.encoder.embeddings(input_ids=label_ids)[0]

 return self.graph_encoder(word_embeddings, attention_mask, labels, embed_label_ids)

def main():
 torch.set_num_threads(2);torch.manual_seed(20261016)
 cfg=BertConfig(vocab_size=64,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,hidden_dropout_prob=.1,attention_probs_dropout_prob=.1,num_labels=24);cfg._attn_implementation='eager'
 child={0:[2,3,4],1:[5,6,7],2:[8,9,10],3:[11,12,13],4:[14,15],5:[16,17,18],6:[19,20],7:[21,22,23]}
 with tempfile.TemporaryDirectory() as td:
  torch.save({i:[i] for i in range(24)},Path(td)/'bert_value_dict.pt');torch.save(child,Path(td)/'slot.pt')
  with patch('model.contrast.AutoTokenizer.from_pretrained',return_value=Tokenizer()),patch('model.graph.AutoTokenizer.from_pretrained',return_value=Tokenizer()):source=ContrastModel(cfg,multi_label=True,graph=True,layer=1,data_path=td,lamb=.05,threshold=.02,tau=1)
 ids=torch.randint(1,64,(4,9));mask=torch.ones_like(ids);mask[0,6:]=0;y=torch.zeros(4,24)
 for i,p in enumerate([[0,2,8],[0,3,11,1,5,16],[1,6],[0,4,14,15]]):y[i,p]=1
 report={}
 for name,fixed in [('legacy_control',False),('corrected_implementation',True)]:
  m=clone(source)
  if not fixed:m.positive_weights=legacy_positive.__get__(m,Hgclr)
  source.zero_grad(set_to_none=True);m.zero_grad(set_to_none=True);source.train();m.train();torch.manual_seed(73);a=source(ids,mask,labels=y);a['loss'].backward();state_a=torch.get_rng_state();torch.manual_seed(73);b=m(ids,mask,labels=y);b['loss'].backward();state_b=torch.get_rng_state()
  err=0.;n=0
  for l,r in [(source.bert,m.encoder),(source.classifier,m.classifier),(source.graph_encoder,m.graph_encoder),(source.contrastive_lossfct.transform,m.projector)]:
   for key,p in l.named_parameters():
    q=dict(r.named_parameters())[key]
    assert (p.grad is None)==(q.grad is None),key
    if p.grad is not None:err=max(err,float((p.grad-q.grad).abs().max()));n+=1
  report[name]={'logits_max_abs_error':float((a['logits']-b['logits']).detach().abs().max()),'loss_abs_error':float((a['loss']-b['loss']).detach().abs()),'gradient_max_abs_error':err,'gradient_parameters':n,'rng_state_identical':bool(torch.equal(state_a,state_b))}
 assert not report['legacy_control']['rng_state_identical']
 assert report['corrected_implementation']['rng_state_identical']
 assert report['corrected_implementation']['gradient_max_abs_error']<1e-6
 out=R/'provenance/dropout_parity.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
