import os,sys,json,random,uuid
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'adapter'),str(R/'tests'),str(R/'upstream')]
import numpy as np,torch
torch.set_num_threads(2)
from test_production import tiny
from training_support import V2Trainer
from transformers import Seq2SeqTrainingArguments,TrainerCallback,set_seed
root=R/'checks'/('resume_cpu_'+uuid.uuid4().hex);root.mkdir()
class Stop(TrainerCallback):
 def on_step_end(self,args,state,control,**kw):
  if state.global_step==2:control.should_training_stop=True
class Dataset(torch.utils.data.Dataset):
 def __len__(self):return 16
 def __getitem__(self,i):
  x=torch.tensor([3+i%12,5,2]);return dict(input_ids=x,attention_mask=torch.ones_like(x),labels=x.clone())
def collate(rows):
 ids=torch.stack([x['input_ids'] for x in rows]);ids=torch.cat([ids,torch.roll(ids,1,1)],0)
 return dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids.clone())
def trainer(name,stop=False):
 set_seed(42);m=tiny()
 # Nonzero dropout exercises actual RNG replay, not deterministic-only weights.
 for mod in m.modules():
  drops=getattr(mod,"lora_dropout",None)
  if isinstance(drops,torch.nn.ModuleDict):
   for adapter_name in list(drops):drops[adapter_name]=torch.nn.Dropout(.1)
 configs=m.peft_config if isinstance(m.peft_config,dict) else {"default":m.peft_config}
 for config in configs.values():config.lora_dropout=.1
 assert any(isinstance(mod,torch.nn.Dropout) and mod.p==.1 for mod in m.modules())
 a=Seq2SeqTrainingArguments(output_dir=str(root/name),no_cuda=True,report_to=[],max_steps=4,
  save_steps=2,logging_steps=1,per_device_train_batch_size=2,gradient_accumulation_steps=2,
  save_safetensors=False,learning_rate=.0002,seed=42,data_seed=42,remove_unused_columns=False,
  disable_tqdm=True,optim='adamw_torch')
 return V2Trainer(model=m,args=a,train_dataset=Dataset(),data_collator=collate,callbacks=[Stop()] if stop else [])
a=trainer('straight');a.train()
b=trainer('first',True);b.train();assert b.state.global_step==2
cp=root/'first/checkpoint-2'
c=trainer('resumed');c.train(resume_from_checkpoint=str(cp));assert c.state.global_step==4
diff=max(float((p-c.model.state_dict()[n]).abs().max()) for n,p in a.model.state_dict().items())
assert diff==0.,diff
oa=a.optimizer.state_dict();oc=c.optimizer.state_dict()
for k,v in oa['state'].items():
 for key,value in v.items():
  if torch.is_tensor(value):torch.testing.assert_close(value,oc['state'][k][key],rtol=0,atol=0)
assert a.lr_scheduler.state_dict()==c.lr_scheduler.state_dict()
a._load_rng_state(str(root/'straight/checkpoint-4'));rng1=(random.random(),np.random.rand(),torch.rand(3))
c._load_rng_state(str(root/'resumed/checkpoint-4'));rng2=(random.random(),np.random.rand(),torch.rand(3))
assert rng1[:2]==rng2[:2];assert torch.equal(rng1[2],rng2[2])
assert not list(root.rglob('pytorch_model.bin'));assert not list(root.rglob('model.safetensors'))
result=dict(passed=True,checkpoint=str(cp),max_parameter_difference=diff,optimizer_equal=True,scheduler_equal=True,rng_equal=True,resumed_global_step=c.state.global_step,adapter_only=True,actual_lora_dropout=.1,no_7b_loaded=True,no_gpu=True)
(R/'checks/RESUME_CPU_COMPLETE_20261020.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
