"""CPU synthetic tests only. No pretrained weights, GPU, actual embeddings or PCA fit."""
import ast, json, math, random, sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, mock_open
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"adapter"),str(ROOT/"upstream")]
from common import masked_mean,dropout_feature,load_mapping,read_vectors,new_path
from quality_pca import fit_common
from model_patch import LlamaRSEmbV2
from transformers import LlamaConfig
from llm.data_processor.llama import llama_train_mask
from llm.data_processor.collator import PairwiseDataCollatorWithPadding

class ToyTokenizer:
    pad_token_id=0; eos_token_id=2
    def encode(self,text,add_special_tokens=False): return [3+ord(x)%20 for x in text]
    def build_inputs_with_special_tokens(self,x): return [1]+x

class Tests(unittest.TestCase):
    def test_pool_padding_and_batch_invariance(self):
        h=torch.tensor([[[1.,3.],[3.,5.],[99.,99.]],[[6.,2.],[77.,77.],[88.,88.]]])
        m=torch.tensor([[1,1,0],[1,0,0]])
        out=masked_mean(h,m)
        torch.testing.assert_close(out,torch.tensor([[2.,4.],[6.,2.]]),rtol=0,atol=0)
        torch.testing.assert_close(out[:1],masked_mean(h[:1,:2],m[:1,:2]))
    def test_pool_fp32_overflow_prevention(self):
        h=torch.full((2,4,8),65000.,dtype=torch.float16)
        v=masked_mean(h,torch.ones(2,4))
        self.assertTrue(torch.isfinite(v).all());self.assertEqual(v.dtype,torch.float32)
        self.assertFalse(torch.isfinite(h.sum(1)).all())
    def test_pool_empty_nonfinite_rejected(self):
        for h,m in [(torch.ones(1,2,3),torch.zeros(1,2)),
                    (torch.full((1,2,3),float("nan")),torch.ones(1,2))]:
            with self.assertRaises((ValueError,FloatingPointError)): masked_mean(h,m)
    def test_augmentation_seed_and_real_attributes(self):
        text="Movie:\ntitle is A; year is 1995; genres are Comedy;"
        a=dropout_feature(text,.4,random.Random(42))
        self.assertEqual(a,dropout_feature(text,.4,random.Random(42)))
        self.assertTrue(all(len([t for t in x.split("\n")[1].split(";") if t.strip()])==2 for x in a))
        self.assertTrue(all("title" in x for x in dropout_feature("M:\ntitle is A;",.99,random.Random(42))))
    def test_prompt_id_validation_and_reject_bad_vectors(self):
        mapping={"1":1,"20":2}
        with patch("builtins.open",mock_open(read_data='{"1":1,"20":1}')):
            with self.assertRaises(ValueError):load_mapping("mapping")
        for rows in [[{"item_id":1,"item":"1","hidden_states":[float("nan"),1]}],
                     [{"item_id":1,"item":"1","hidden_states":[0,0]}],
                     [{"item_id":1,"item":"20","hidden_states":[1,1]}],
                     [{"item_id":1,"item":"1","hidden_states":[1,1]}]*2]:
            with patch("builtins.open",mock_open(read_data="\n".join(json.dumps(x) for x in rows))):
                with self.assertRaises(ValueError):read_vectors("mock",mapping,2)
    def test_common_pca_and_padding_synthetic_only(self):
        rng=np.random.default_rng(42); a=np.vstack([np.zeros((1,4)),rng.normal(size=(8,4))])
        model,b,r=fit_common(a,a.copy(),list(range(1,9)),3)
        np.testing.assert_array_equal(b,r);np.testing.assert_array_equal(b[0],np.zeros(3))
        projected=(a[1:]-model.mean_)@model.components_.T
        np.testing.assert_allclose(projected,b[1:],rtol=1e-5,atol=1e-6)
    def test_real_mask_forward_and_odd_inference_tiny_model(self):
        cfg=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                        num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=32,pad_token_id=0)
        torch.manual_seed(42);model=LlamaRSEmbV2(cfg,pool_type="avg",tau=3.).eval()
        ids=torch.tensor([[3,4,2,0],[5,2,0,0],[8,9,10,2]])
        mask=ids.ne(0)
        out=model(input_ids=ids,attention_mask=mask)
        self.assertEqual(tuple(out.logits.shape),(3,16))
        single=model(input_ids=ids[:1,:3],attention_mask=mask[:1,:3])
        torch.testing.assert_close(out.logits[:1],single.logits,rtol=2e-5,atol=2e-5)
        pair=ids[[0,1,0,1]];pm=mask[[0,1,0,1]]
        loss=model(input_ids=pair,attention_mask=pm,labels=pair).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(model.tau.grad).all())
    def test_contrastive_fp32_without_formula_change(self):
        from llm.llama import Contrastive_Loss
        x=torch.full((2,4096),10.,dtype=torch.float16)
        self.assertFalse(torch.isfinite(x@x.T).all())
        loss=Contrastive_Loss(torch.tensor([3.]))(x.float(),x.float())
        self.assertTrue(torch.isfinite(loss))
    def test_official_processor_and_pair_order(self):
        tok=ToyTokenizer();args=SimpleNamespace(max_source_length=16,max_target_length=4,
                                    ignore_pad_token_for_loss=True,dropout_ratio=.4)
        text="Movie:\ntitle A; year B; genre C;"
        with patch("llm.data_processor.llama.dropout_feature",dropout_feature):
            p=llama_train_mask(args,None,tok)({"input":[text,text+" "],"target":["",""]})
        items=[{k:torch.tensor(v[i]) for k,v in p.items()} for i in range(2)]
        batch=PairwiseDataCollatorWithPadding(tok)(items)
        self.assertEqual(batch["input_ids"].shape[0],4)
        for i,field in enumerate(["chosen_ids","chosen_ids","rejected_ids","rejected_ids"]):
            torch.testing.assert_close(batch["input_ids"][i,:len(items[i%2][field])],items[i%2][field])
        self.assertTrue((batch["attention_mask"].sum(1)>0).all())
    def test_write_boundary(self):
        with self.assertRaises(ValueError):new_path("/tmp/unauthorized")
        with self.assertRaises(FileExistsError):new_path(ROOT/"adapter"/"common.py")
    def test_trainable_tau_stays_fp32(self):
        cfg=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                        num_hidden_layers=1,num_attention_heads=2,pad_token_id=0)
        model=LlamaRSEmbV2(cfg,pool_type="avg",tau=3.).half()
        self.assertEqual(model.tau.dtype,torch.float32)
        self.assertEqual(model.model.embed_tokens.weight.dtype,torch.float16)
    def test_mean_imputation_then_centering_explains_zero(self):
        # Algebra demonstration only; V2 never performs mean imputation.
        good=np.array([[1.,4.],[3.,2.],[5.,3.]],dtype=np.float64)
        mean=good.mean(0); joined=np.vstack([good,mean,mean])
        np.testing.assert_allclose((joined-joined.mean(0))[-2:],0,atol=1e-15)
    def test_static_all_sources(self):
        for path in (ROOT/"adapter").glob("*.py"): ast.parse(path.read_text())
        # Every heavy CLI returns before LLM/PCA work unless explicitly executed.
        for name in ["run_scft.py","run_encode.py","quality_pca.py"]:
            self.assertIn("if not a.execute", (ROOT/"adapter"/name).read_text())

if __name__=="__main__":unittest.main(verbosity=2)
