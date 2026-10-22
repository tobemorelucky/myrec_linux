"""1024 policy tests on CPU only, no pretrained model."""
import json,sys,unittest,random,functools,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"adapter"),str(ROOT/"upstream"),str(ROOT/"tests")]
from common import encode_prompt,dropout_feature
from training_support import FixedAdmissionSampler,V2TrainMask,V2Trainer
from test_production import Token,tiny
class Tok(Token):
    pad_token_id=0
    def encode(self,text=None,add_special_tokens=False):return super().encode(text,add_special_tokens=add_special_tokens)
    def build_inputs_with_special_tokens(self,ids):return ids+[2]
class FullText(unittest.TestCase):
    def test_no_160_field_budget(self):
        text="Movie:\ntitle is A; overview is "+"a"*800+";"
        ids,st=encode_prompt(text,Tok(),1024,{},strict=True)
        self.assertEqual(len(ids),len(text)+1);self.assertFalse(st["field_tokens_clipped"])
    def test_overlength_rejected_not_sliced(self):
        with self.assertRaises(ValueError):encode_prompt("M:\n"+"x"*1021,Tok(),1024,{},strict=True)
    def test_eos_boundary(self):
        self.assertEqual(len(encode_prompt("M:\n"+"x"*1020,Tok(),1024,{},strict=True)[0]),1024)
    def test_hidden_field_cap_rejected(self):
        with self.assertRaises(ValueError):
            encode_prompt("Movie:\ntitle is A; overview is "+"x"*200,Tok(),1024,{"overview":160},strict=True)
    def test_fixed_sampler_exact_order(self):
        order=list(range(383,-1,-1));s=FixedAdmissionSampler(order,3706)
        self.assertEqual(list(s),order);self.assertEqual(len(s)//16//8,3)
        with self.assertRaises(ValueError):FixedAdmissionSampler([3706],3706)
    def test_full_train_processor_same_body_eos(self):
        import llm.data_processor.llama as proc
        from types import SimpleNamespace
        saved=proc.dropout_feature
        proc.dropout_feature=functools.partial(dropout_feature,rng=random.Random(42))
        try:
            class F(V2TrainMask):fail_on_truncation=True
            f=F(SimpleNamespace(max_source_length=1024,max_target_length=196,dropout_ratio=.4,ignore_pad_token_for_loss=True),None,Tok())
            text="Movie:\ntitle is A; release year is 1995; genres are Comedy; overview is "+"x"*700+";"
            out=f({"input":[text],"target":[""]})
            self.assertTrue(all(len(out[k][0])<=1024 for k in ("chosen_ids","rejected_ids")))
            with self.assertRaises(ValueError):f({"input":[text+"x"*1024],"target":[""]})
        finally:proc.dropout_feature=saved
    def test_new_encode_training_policy_matches(self):
        for name in ["scft_rich_full1024_pending.json","encode_base_full1024.json","encode_rich_full1024.json"]:
            c=json.loads((ROOT/"configs"/name).read_text())
            self.assertEqual(c["max_source_length"],1024);self.assertEqual(c["field_caps"],{})
            self.assertTrue(c["fail_on_truncation"])
    def test_old_frozen_configs_preserved(self):
        c=json.loads((ROOT/"configs/scft_rich_frozen.json").read_text())
        self.assertEqual(c["max_source_length"],256);self.assertEqual(c["field_caps"]["overview"],160)
    def test_admission_counts_actual_optimizer_calls(self):
        import torch
        from transformers import Seq2SeqTrainingArguments
        class T(V2Trainer):
            admission_indices=[0]
            admission_optimizer_updates=0
        args=Seq2SeqTrainingArguments(output_dir=str(ROOT/"checks"/"fulltext_tests"/uuid.uuid4().hex),report_to=[])
        trainer=T(model=tiny(),args=args);trainer.create_optimizer()
        for _ in range(3):
            trainer.optimizer.zero_grad()
            loss=sum(p.float().sum() for p in trainer.model.parameters() if p.requires_grad)
            loss.backward();trainer.optimizer.step()
        self.assertEqual(T.admission_optimizer_updates,3)

if __name__=="__main__":unittest.main(verbosity=2)
