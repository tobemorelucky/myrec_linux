"""Minimal corrections to pinned official LLMEmb; transformer/loss are reused."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"upstream"))
import torch
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from llm.llama import LlamaRSEmb as Original, Contrastive_Loss
from common import masked_mean

class LlamaRSEmbV2(Original):
    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        # Existing local weights only; never fetch a duplicate base or select the bin copy.
        kwargs["local_files_only"]=True
        kwargs["use_safetensors"]=True
        return super().from_pretrained(*args,**kwargs)
    def half(self):
        super().half()
        # Official main casts the base to half before PEFT. Keep learned tau
        # FP32 for GradScaler and small Adam updates.
        self.tau.data = self.tau.data.float()
        return self
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=False,
                output_attentions=False, output_hidden_states=False, return_dict=True):
        if attention_mask is None:
            raise ValueError("Explicit real-token mask required")
        out=self.model(input_ids=input_ids,attention_mask=attention_mask,
                       position_ids=position_ids,past_key_values=past_key_values,
                       inputs_embeds=inputs_embeds,use_cache=False,
                       output_attentions=output_attentions,
                       output_hidden_states=False,return_dict=True)
        if self.pool_type!="avg": raise ValueError("V2 protocol fixes masked FP32 average")
        pooled=masked_mean(out.last_hidden_state,attention_mask)
        loss=None
        if labels is not None:
            if len(pooled)%2 or len(pooled)<4:
                raise ValueError("SCFT needs >=2 original items and paired augmented views")
            if not torch.isfinite(self.tau).all() or self.tau.item()<=1e-6:
                raise FloatingPointError("SCFT temperature invalid")
            x,y=pooled.chunk(2)
            # Exact public loss algebra, FP32 matmuls instead of FP16 overflow.
            with torch.autocast(device_type=pooled.device.type,enabled=False):
                loss=Contrastive_Loss(self.tau.float())(x.float(),y.float())
            if not torch.isfinite(loss): raise FloatingPointError("SCFT loss nonfinite")
        return SequenceClassifierOutputWithPast(loss=loss,logits=pooled,hidden_states=pooled,
                                                 attentions=out.attentions)
