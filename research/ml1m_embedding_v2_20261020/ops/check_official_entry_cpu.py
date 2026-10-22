"""CPU official-entry integration: tiny random model, real tokenizer/data, ZERO updates."""
import ast,functools,json,random,sys,types,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"adapter"),str(ROOT/"upstream")]
from common import load_config,dropout_feature,json_new,sha256
from training_support import V2TrainMask,V2Trainer,local_scft_dataset
from model_patch import LlamaRSEmbV2
from transformers import AutoTokenizer,LlamaConfig
import llm.data_processor.llama as proc
c=load_config(ROOT/"configs/scft_rich_full1024_frozen.json")
folder=ROOT/"checks"/("official_entry_cpu_"+uuid.uuid4().hex);folder.mkdir()
alias=folder/"train_input.json";alias.write_bytes(Path(c["prompt"]).read_bytes())
assert sha256(alias)==sha256(c["prompt"])
tree=ast.parse((ROOT/"upstream/main_llm.py").read_text())
tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(
 isinstance(t,ast.Subscript) and isinstance(t.value,ast.Attribute) and t.value.attr=="environ" and
 isinstance(t.slice,ast.Constant) and t.slice.value=="CUDA_VISIBLE_DEVICES" for t in n.targets))]
mod=types.ModuleType("official_entry_cpu");mod.__file__=str(ROOT/"upstream/main_llm.py")
exec(compile(tree,mod.__file__,"exec"),mod.__dict__)
class Tiny(LlamaRSEmbV2):
    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        return cls(LlamaConfig(vocab_size=32000,hidden_size=16,intermediate_size=32,
              num_hidden_layers=1,num_attention_heads=2,pad_token_id=0,max_position_embeddings=1024),
              pool_type="avg",tau=3.)
    def cuda(self,*args,**kwargs):return self
    def half(self):return self
class LocalTokenizer:
    @staticmethod
    def from_pretrained(*args,**kwargs):
        kwargs.update(local_files_only=True,trust_remote_code=False)
        return AutoTokenizer.from_pretrained(*args,**kwargs)
facts={}
class NoUpdateTrainer(V2Trainer):
    def train(self,*args,**kwargs):
        self.create_optimizer()
        b=next(iter(self.get_train_dataloader()))
        assert b["input_ids"].shape[0]==32 and b["input_ids"].shape[1]<=1024
        assert any(p.requires_grad for n,p in self.model.named_parameters() if "lora_" in n)
        facts.update(rows=len(self.train_dataset),batch_shape=list(b["input_ids"].shape),
            optimizer_constructed=True,optimizer_steps=0,model_is_tiny=True,actual_7b_loaded=False)
        return None
    def save_state(self):facts["official_post_train_path_reached"]=True
class Mask(V2TrainMask):fail_on_truncation=True
mod.LlamaRSEmb=Tiny;mod.AutoTokenizer=LocalTokenizer;mod.MedRecTrainer=NoUpdateTrainer
loader=mod.load_dataset
mod.load_dataset=lambda *a,**kw:local_scft_dataset(loader,*a,**kw)
mod.llama_train_mask=Mask
proc.dropout_feature=functools.partial(dropout_feature,rng=random.Random(42))
sys.argv=["cpu_official","--do_train","--no_cuda","true","--model_name_or_path",c["model_path"],
 "--train_file",str(alias),"--cache_dir",str(folder/"cache"),"--output_dir",str(folder/"out"),
 "--max_source_length","1024","--max_target_length","196","--per_device_train_batch_size","16",
 "--gradient_accumulation_steps","8","--max_steps","3","--learning_rate","0.0002",
 "--lora_rank","8","--lora_alpha","32","--lora_dropout","0.1","--trainable",c["target_modules"],
 "--pool_type","avg","--dropout_ratio","0.4","--seed","42","--report_to","none",
 "--preprocessing_num_workers","1","--dataloader_drop_last","false","--save_safetensors","false"]
mod.train()
json_new(folder/"COMPLETE.json",facts)
print(json.dumps(facts,indent=2))
