"""Tiny CPU only: no pretrained weights or GPU."""
import json,sys,unittest,uuid
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"adapter"),str(ROOT/"upstream")]
from numerical_admission import LlamaRSEmbBF16,FirstNonfinite,precision_report,finite_tensor,blocks,prepare_training_probe
from transformers import LlamaConfig
from llm.peft import LoraConfig
from llm.lora_cls import PeftModelForCLS
class Numerical(unittest.TestCase):
    def tiny(self):
        cfg=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                        num_attention_heads=2,max_position_embeddings=32,pad_token_id=0)
        base=LlamaRSEmbBF16(cfg,pool_type="avg",tau=3.).half()
        model=PeftModelForCLS(base,LoraConfig(r=8,lora_alpha=32,lora_dropout=.1,
                 target_modules=["q_proj","k_proj","v_proj","o_proj","down_proj","gate_proj","up_proj"],
                 task_type="SEQ_CLS"))
        for n,p in model.named_parameters():
            if n.endswith("tau"):p.requires_grad=True
        precision_report(model,fix_trainable=True)
        return model
    def test_official_half_call_really_uses_bf16(self):
        counts=precision_report(self.tiny())
        self.assertIn("torch.bfloat16",counts);self.assertIn("torch.float32",counts)
        self.assertNotIn("torch.float16",counts)
    def test_bf16_train_forward_backward_original_loss(self):
        m=self.tiny().train()
        ids=torch.tensor([[3,4,2,0],[5,6,7,2],[3,4,8,2],[5,9,2,0]])
        with torch.autocast("cpu",dtype=torch.bfloat16):
            out=m(input_ids=ids,attention_mask=(ids!=0).long(),labels=torch.zeros(4))
        self.assertEqual(out.loss.dtype,torch.float32);self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        grads=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads);self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.float().norm()) for g in grads),0.)
    def test_first_bad_matmul_is_recorded(self):
        class Bad(torch.nn.Module):
            def forward(self,x):return x@x
        path=ROOT/"checks"/"numerical_cpu"/(uuid.uuid4().hex+".json")
        with self.assertRaises(FloatingPointError):
            with FirstNonfinite(Bad(),path):
                Bad()(torch.full((2,2),60000.,dtype=torch.float16))
        result=json.loads(path.read_text())
        self.assertIn("aten.mm.",result["first_observed_failure"])
        self.assertEqual(result["operator_inputs"][0]["dtype"],"torch.float16")
        self.assertGreater(result["outputs"][0]["posinf"],0)
    def test_finite_masked_forward_not_false_alarm(self):
        m=self.tiny().eval();path=ROOT/"checks"/"numerical_cpu"/(uuid.uuid4().hex+".json")
        ids=torch.tensor([[3,4,2,0],[5,6,7,2]])
        with FirstNonfinite(m,path):
            with torch.autocast("cpu",dtype=torch.bfloat16):
                out=m(input_ids=ids,attention_mask=(ids!=0).long(),labels=None)
        self.assertTrue(torch.isfinite(out.logits).all());self.assertFalse(path.exists())
    def test_chunked_finiteness_and_noncontiguous_views(self):
        x=torch.arange(8*16,dtype=torch.float32).reshape(8,16).T
        self.assertTrue(finite_tensor(x));self.assertTrue(all(t.numel()<=8 for t in blocks(x,8)))
        for value in (float("nan"),float("inf"),-float("inf")):
            z=x.clone();z[3,2]=value
            self.assertFalse(finite_tensor(z))
    def test_exception_location_saved_without_gpu_allocations(self):
        class Broken(torch.nn.Linear):
            def forward(self,x):raise RuntimeError("synthetic interruption")
        path=ROOT/"checks"/"numerical_cpu"/(uuid.uuid4().hex+".json")
        path.parent.mkdir(parents=True,exist_ok=True)
        model=Broken(2,2)
        with self.assertRaises(RuntimeError):
            with FirstNonfinite(model,path):model(torch.ones(2,2))
        result=json.loads(path.with_name(path.stem+"_EXCEPTION.json").read_text())
        self.assertEqual(result["error_type"],"RuntimeError")
        self.assertEqual(result["module_input_metadata"][-1]["tensors"][0]["shape"],[2,2])
    def test_probe_enables_real_training_checkpoint_and_dropout(self):
        m=self.tiny()
        m.eval()
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()
        self.assertFalse(m.training)
        result=prepare_training_probe(m)
        self.assertTrue(result["training"]);self.assertTrue(result["gradient_checkpointing"])
        self.assertTrue(all(x.training for x in m.modules() if isinstance(x,torch.nn.Dropout)))
        for core in m.modules():
            if type(core).__name__=="LlamaModel":self.assertTrue(core.training and core.gradient_checkpointing)
    def test_new_config_does_not_change_protocol(self):
        old=json.loads((ROOT/"configs/scft_rich_full1024_frozen.json").read_text())
        new=json.loads((ROOT/"configs/scft_rich_full1024_bf16_numeric.json").read_text())
        for key in ("batch_size","gradient_accumulation_steps","max_source_length","learning_rate",
                    "max_steps","seed","prompt","expected_prompt_sha256","target_modules","field_caps"):
            self.assertEqual(old[key],new[key])
        self.assertEqual(new["precision"],"bf16");self.assertTrue(new["numerical_admission"])
if __name__=="__main__":unittest.main(verbosity=2)
