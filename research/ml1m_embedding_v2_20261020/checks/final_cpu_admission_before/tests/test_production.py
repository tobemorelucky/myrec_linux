"""Finite CPU checks only; never loads base weights or runs Trainer.train."""
import json,sys,unittest,uuid
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"adapter"),str(ROOT/"upstream")]
from common import (dropout_feature,encode_prompt,table_quality,restore_extras,verify_adapter_loaded,
                    new_path)
from model_patch import LlamaRSEmbV2
from transformers import LlamaConfig,Seq2SeqTrainingArguments
from llm.peft import LoraConfig
from llm.lora_cls import PeftModelForCLS
from training_support import V2Trainer,save_adapter_complete

class Token:
    eos_token_id=2
    def encode(self,s,add_special_tokens=False):return [3+ord(c)%128 for c in s]
    def decode(self,ids,skip_special_tokens=True):return "".join(chr(i-3) for i in ids)

def tiny():
    cfg=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,pad_token_id=0,max_position_embeddings=32)
    base=LlamaRSEmbV2(cfg,pool_type="avg",tau=3.)
    m=PeftModelForCLS(base,LoraConfig(r=2,lora_alpha=4,target_modules=["q_proj","v_proj"],
                                    lora_dropout=0.,task_type="SEQ_CLS"))
    for n,p in m.named_parameters():
        if "tau" in n:p.requires_grad=True
    return m

class Production(unittest.TestCase):
    def test_original_epoch_accumulation_boundaries(self):
        from torch.utils.data import DataLoader
        loader=DataLoader(range(3706),batch_size=16,drop_last=False)
        self.assertEqual(len(loader),232)
        self.assertEqual(len(loader)%8,0)
        self.assertEqual(len(list(loader)[-1]),10)
        self.assertIn('"--dataloader_drop_last","false"',(ROOT/"adapter"/"run_scft.py").read_text())
    def test_tau_has_no_effect_on_inference(self):
        m=tiny().eval();ids=torch.tensor([[3,4,2],[5,6,2]]);mask=torch.ones_like(ids)
        with torch.no_grad():
            first=m(input_ids=ids,attention_mask=mask,labels=None).logits.clone()
            for n,p in m.named_parameters():
                if "tau" in n:p.fill_(-19.)
            second=m(input_ids=ids,attention_mask=mask,labels=None).logits
        torch.testing.assert_close(first,second,rtol=0,atol=0)
    def test_title_identity_kept_and_two_views(self):
        import random
        for seed in range(20):
            views=dropout_feature("Movie:\ntitle is A; release year is 1995; genres are Comedy;",.4,random.Random(seed))
            self.assertTrue(all("title is A" in v for v in views))
            self.assertTrue(all(len([x for x in v.split("\n")[1].split(";") if x.strip()])==2 for v in views))
    def test_truncation_is_bounded_and_counted(self):
        text="Movie:\ntitle is X; themes is "+"a"*60+"; plot is "+"b"*100+";"
        ids,stats=encode_prompt(text,Token(),80,{"themes":20,"plot":30})
        self.assertLessEqual(len(ids),80);self.assertEqual(ids[-1],2)
        self.assertIn("themes",stats["field_tokens_clipped"])
        self.assertIn("plot",stats["field_tokens_clipped"])
    def test_quality_blocks_nearzero_and_duplicates(self):
        for a in [np.array([[0.,0.],[1e-8,0.],[1.,1.]]),
                  np.array([[0.,0.],[1.,1.],[1.,1.]])]:
            with self.assertRaises(ValueError):table_quality(a)
    def test_adapter_save_load_extra_and_no_full_backbone(self):
        m=tiny();root=ROOT/"checks"/"production_tests"/("save_"+uuid.uuid4().hex)
        save_adapter_complete(m,root)
        self.assertTrue((root/"adapter_model.bin").is_file())
        self.assertFalse((root/"pytorch_model.bin").exists())
        self.assertTrue((root/"v2_rng_state.pth").is_file())
        self.assertGreater(verify_adapter_loaded(m,root),0)
        for n,p in m.named_parameters():
            if "tau" in n:p.data.fill_(33.)
        restore_extras(m,root)
        for n,p in m.named_parameters():
            if "tau" in n:self.assertEqual(p.item(),3.)
    def test_trainer_checkpoint_contains_training_state(self):
        from transformers.trainer_callback import TrainerState
        m=tiny();folder=ROOT/"checks"/"production_tests"/("trainer_"+uuid.uuid4().hex)
        args=Seq2SeqTrainingArguments(output_dir=str(folder),no_cuda=True,report_to=[],
                                    save_safetensors=False,learning_rate=.0002)
        t=V2Trainer(model=m,args=args);t.create_optimizer();t.create_scheduler(10)
        t.state=TrainerState(global_step=1)
        t._save_checkpoint(m,trial=None)
        p=folder/"checkpoint-1"
        for name in ("adapter_model.bin","v2_extra_trainable.pt","training_args.json",
                     "optimizer.pt","scheduler.pt","rng_state.pth","trainer_state.json"):
            self.assertTrue((p/name).is_file(),name)
        self.assertFalse((p/"pytorch_model.bin").exists())
    def test_original_scft_has_finite_task_gradients(self):
        m=tiny().train();ids=torch.tensor([[3,4,2],[5,6,2],[3,7,2],[5,8,2]])
        loss=m(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids).loss
        loss.backward()
        grads=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads);self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.norm()) for g in grads),0)
if __name__=="__main__":unittest.main(verbosity=2)
