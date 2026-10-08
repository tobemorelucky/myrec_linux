"""CPU synthetic tests. No datasets, checkpoints, logging files or temporary assets."""
import argparse
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from models.BaseModel import GeneralModel
from models.sequential.LLMMIRecASPCF import LLMMIRecASPCF
from models.sequential.LLMMIRecASPCFPoMBridge import LLMMIRecASPCFPoMBridge
from models.sequential.PoMRec import MultiInterestExtractor
from models.sequential.pomrec_backbone import PoMRecInterestBackbone

torch.set_num_threads(1)


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = torch.randn(33, 1536, generator=torch.Generator().manual_seed(17))
        cls.table[0] = 0
        cls.corpus = SimpleNamespace(n_items=33, n_users=8)

    def args(self, mode="clean", dispersion=1.0, dropout=0.1):
        parser = LLMMIRecASPCFPoMBridge.parse_model_args(argparse.ArgumentParser())
        args = parser.parse_args([])
        args.bridge_mode, args.bridge_lambda_disp = mode, dispersion
        args.device, args.model_path = torch.device("cpu"), "unused-no-write.pt"
        args.llm_emb_path, args.history_max = "synthetic://in-memory", 20
        args.num_neg, args.dropout, args.relation_sample_size = 2, dropout, 5
        return args

    def model(self, mode="clean", dispersion=1.0, seed=42, original=False):
        torch.manual_seed(seed)
        with patch("models.sequential.LLMMIRecASPCF.load_llm_table", return_value=self.table):
            cls = LLMMIRecASPCF if original else LLMMIRecASPCFPoMBridge
            return cls(self.args(mode, dispersion), self.corpus)

    @staticmethod
    def feed():
        return {"history_items": torch.tensor([[1,2,3,4,0], [5,6,0,0,0], [7,8,9,0,0]]),
                "lengths": torch.tensor([4,2,3]),
                "item_id": torch.tensor([[10,11,12], [13,14,15], [16,17,18]])}

    def exact(self, a, b):
        self.assertTrue(torch.equal(a, b), f"not exact; max error={(a-b).abs().max().item()}")

    def test_c0_strict_state_predictions_loss_gradients(self):
        original = self.model(original=True)
        rng_original = torch.get_rng_state()
        clean = self.model()
        self.exact(rng_original, torch.get_rng_state())
        self.assertEqual(original.state_dict().keys(), clean.state_dict().keys())
        for key, value in original.state_dict().items():
            self.exact(value, clean.state_dict()[key])
        clean.load_state_dict(original.state_dict(), strict=True)
        for training in (False, True):
            original.train(training); clean.train(training)
            original.zero_grad(); clean.zero_grad()
            torch.manual_seed(71)
            a = original(self.feed(), return_intermediate=True)
            loss_a = original.loss(a)
            torch.manual_seed(71)
            b = clean(self.feed(), return_intermediate=True)
            loss_b = clean.loss(b)
            for key in ("prediction","interest_vectors","interest_weights","attention_maps","user_vector"):
                self.exact(a[key], b[key])
            self.exact(loss_a, loss_b)
            if training:
                self.exact(a["_relation_ids"], b["_relation_ids"])
                self.exact(a["loss_relation"], b["loss_relation"])
            loss_a.backward(); loss_b.backward()
            for (_, p), (_, q) in zip(original.named_parameters(), clean.named_parameters()):
                self.exact(p.grad, q.grad)

    def test_shared_initialization_and_rng(self):
        original = self.model(original=True)
        reference_rng = torch.get_rng_state()
        reference = {k:v.clone() for k,v in original.state_dict().items()
                     if k.startswith(("item_encoder.", "position_emb."))}
        for mode in ("pom_centrality","pom_full"):
            model = self.model(mode)
            self.exact(reference_rng, torch.get_rng_state())
            for key, value in reference.items():
                self.exact(value, model.state_dict()[key])
        # The unchanged asset loader is called exactly once with parent arguments.
        with patch("models.sequential.LLMMIRecASPCF.load_llm_table", return_value=self.table) as loader:
            LLMMIRecASPCFPoMBridge(self.args("pom_full"), self.corpus)
            loader.assert_called_once_with("synthetic://in-memory", expected_rows=33)

    def test_c2_zero_identical_to_c1_and_no_sqrt(self):
        c1, c2 = self.model("pom_centrality"), self.model("pom_full", 0.0)
        self.assertEqual(c1.state_dict().keys(), c2.state_dict().keys())
        for k,v in c1.state_dict().items():
            self.exact(v, c2.state_dict()[k])
        c2.load_state_dict(c1.state_dict(), strict=True)
        for training in (False,True):
            c1.train(training); c2.train(training)
            torch.manual_seed(89)
            a = c1(self.feed(), True)
            la = c1.loss(a)
            torch.manual_seed(89)
            with patch("torch.sqrt", wraps=torch.sqrt) as sqrt:
                b = c2(self.feed(), True)
                # Encoder uses sqrt too; check backbone independently below.
            lb = c2.loss(b)
            self.exact(a["prediction"], b["prediction"])
            self.exact(la, lb)
            self.assertIsNone(a["dispersion"]); self.assertIsNone(b["dispersion"])
            c1.zero_grad(); c2.zero_grad()
            la.backward(); lb.backward()
            for (_,p),(_,q) in zip(c1.named_parameters(), c2.named_parameters()):
                self.exact(p.grad, q.grad)
        with patch("torch.sqrt", side_effect=AssertionError("sqrt must be skipped")):
            c1.backbone(torch.randn(2,3,64), torch.ones(2,3,dtype=torch.bool), torch.tensor([3,3]))

    def test_original_pomrec_attention_mean_rms_reference(self):
        torch.manual_seed(31)
        old = MultiInterestExtractor(k=4,item_num=33,emb_size=64,attn_size=8,max_his=5,
                                     prompt_num=3,lamb=1.0,use_llmemb=0)
        new = PoMRecInterestBackbone(64,4,8,3,2,1.0).double().eval()
        old = old.double().eval()
        with torch.no_grad():
            old.p_embeddings.weight.zero_()
            for name in ("prompt1","prompt2","W1","W3"):
                getattr(new,name).load_state_dict(getattr(old,name).state_dict())
            new.W2.weight.copy_(old.W2.weight); new.W4.weight.copy_(old.W4.weight)
            old.W2.bias.copy_(torch.randn(4)); old.W4.bias.fill_(2.3)
        feed = self.feed()
        H = old.i_embeddings(feed["history_items"])
        result = new(H, feed["history_items"]>0, feed["lengths"], True)
        ref_V, ref_g = old(feed["history_items"], feed["lengths"])
        torch.testing.assert_close(result["interest_vectors"], ref_V, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(result["distribution_vector"], ref_g, rtol=1e-12, atol=1e-12)
        mask = torch.cat([feed["history_items"]>0, torch.ones(3,5,dtype=torch.bool)],1)
        prompts = torch.cat([new.prompt_pad,new.prompt1.weight],0)
        ext = torch.cat([H, prompts[None].expand(3,-1,-1)],1)
        logits = new.W2(new.W1(ext).tanh()).transpose(1,2)
        attn = logits.masked_fill(~mask[:,None,:], -torch.inf).softmax(-1)
        center = torch.zeros_like(result["centrality"])
        var = torch.zeros_like(center)
        for b in range(3):
            for k in range(4):
                for j in range(10):
                    center[b,k] += attn[b,k,j]*ext[b,j]
                for j in range(10):
                    var[b,k] += attn[b,k,j]*(ext[b,j]-center[b,k]).square()
        torch.testing.assert_close(result["centrality"],center,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(result["dispersion"],var.sqrt(),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(result["interest_weights"],
                                  new.proj(result["distribution_vector"]).softmax(-1),rtol=0,atol=0)

    def test_one_encoder_no_unused_tables_parameters_and_counts(self):
        for mode in ("clean","pom_centrality","pom_full"):
            model = self.model(mode)
            self.assertEqual(model.loss.__func__, LLMMIRecASPCF.loss)
            self.assertEqual(model._compute_relation_loss.__func__, LLMMIRecASPCF._compute_relation_loss)
            tables = [name for name,module in model.named_modules()
                      if isinstance(module,torch.nn.Embedding) and module.num_embeddings==33]
            self.assertEqual(tables, ["item_encoder.complement_id_emb"])
            calls=[]
            handle=model.item_encoder.register_forward_pre_hook(lambda module,args: calls.append(args[0].clone()))
            model.eval(); model(self.feed())
            handle.remove()
            self.assertEqual(len(calls),2)
            self.exact(calls[0],self.feed()["history_items"])
            self.exact(calls[1],self.feed()["item_id"])
            if mode!="clean":
                self.assertFalse(hasattr(model,"extractor")); self.assertFalse(hasattr(model,"aggregator"))
                self.assertEqual(set(dict(model.backbone.named_children())),
                                 {"prompt1","prompt2","W1","W2","W3","W4","proj"})
                self.assertEqual(sum(p.numel() for p in model.backbone.parameters()),5884)
            expected=64*33+(168646 if mode=="clean" else 157246)
            self.assertEqual(model.count_variables(),expected)
        for N in (12102,3707):
            args=self.args("pom_full")
            table=torch.zeros(N,1536)
            for mode in ("clean","pom_centrality","pom_full"):
                args.bridge_mode=mode
                with patch("models.sequential.LLMMIRecASPCF.load_llm_table",return_value=table):
                    model=LLMMIRecASPCFPoMBridge(args,SimpleNamespace(n_items=N,n_users=8))
                expected=64*N+(168646 if mode=="clean" else 157246)
                self.assertEqual(model.count_variables(),expected)
                print(f"PARAMETERS N={N} mode={mode}: {expected}")

    def test_padding_and_reverse_positions(self):
        for mode in ("clean","pom_centrality","pom_full"):
            model=self.model(mode).eval()
            feed=self.feed()
            seen=[]
            hook=model.position_emb.register_forward_pre_hook(lambda module,args: seen.append(args[0].clone()))
            score=model(feed)["prediction"]
            hook.remove()
            self.exact(seen[0],torch.tensor([[4,3,2,1,0],[2,1,0,0,0],[3,2,1,0,0]]))
            with torch.no_grad():
                model.position_emb.weight[0].fill_(1000)
                model.item_encoder.complement_id_emb.weight[0].fill_(1000)
            self.exact(score,model(feed)["prediction"])
            extended=copy.deepcopy(feed)
            extended["history_items"]=F.pad(feed["history_items"],(0,4))
            torch.testing.assert_close(score,model(extended)["prediction"],rtol=1e-6,atol=1e-7)
            if mode!="clean":
                mask=feed["history_items"]>0
                H=torch.randn(3,5,64)
                altered=H.clone(); altered[~mask]=float("nan")
                a=model.backbone(H,mask,feed["lengths"],True)
                b=model.backbone(altered,mask,feed["lengths"],True)
                self.exact(a["interest_vectors"],b["interest_vectors"])

    def test_finite_gradients_and_no_idle_parameters(self):
        for mode in ("clean","pom_centrality","pom_full"):
            model=self.model(mode).train()
            out=model(self.feed(),True)
            loss=model.loss(out)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
                self.assertGreater(p.grad.abs().sum().item(),0,name)
        # Degenerate identical tokens: exactly zero std and finite derivative.
        model=PoMRecInterestBackbone(4,2,3,3,1,1.0)
        with torch.no_grad():
            model.prompt1.weight.fill_(1)
        H=torch.ones(2,3,4,requires_grad=True)
        out=model(H,torch.ones(2,3,dtype=torch.bool),torch.tensor([3,3]),True)
        torch.testing.assert_close(out["dispersion"],torch.zeros_like(out["dispersion"]),atol=1e-7,rtol=0)
        out["interest_vectors"].sum().backward()
        self.assertTrue(torch.isfinite(H.grad).all())
        var=torch.tensor([0.0,1.0,4.0],requires_grad=True)
        std=model._zero_safe_sqrt(var)
        self.exact(std,torch.tensor([0.0,1.0,2.0]))
        std.sum().backward()
        self.exact(var.grad,torch.tensor([0.0,0.5,0.25]))

    def test_eval_candidate_permutation_and_set_independence(self):
        for mode in ("clean","pom_centrality","pom_full"):
            model=self.model(mode).eval()
            feed=self.feed()
            a=model(feed,True)
            perm=torch.tensor([2,0,1])
            changed=copy.deepcopy(feed); changed["item_id"]=feed["item_id"][:,perm]
            b=model(changed,True)
            self.exact(b["prediction"],a["prediction"][:,perm])
            self.exact(a["user_vector"],b["user_vector"])
            changed["item_id"]=torch.cat([feed["item_id"],torch.tensor([[21],[22],[23]])],1)
            c=model(changed,True)
            self.exact(a["prediction"],c["prediction"][:,:3])
            self.exact(a["user_vector"],c["user_vector"])
            self.assertNotIn("_relation_ids",a)

    def test_relation_math_sampling_and_loss_composition(self):
        clean=self.model("clean"); bridge=self.model("pom_full")
        self.assertIs(LLMMIRecASPCFPoMBridge.loss,LLMMIRecASPCF.loss)
        for model in (clean,bridge):
            ids=torch.tensor([1,2,3,4,5])
            t=F.normalize(self.table[ids,:512],dim=-1,eps=1e-8)
            s=F.normalize(model.item_encoder.semantic_branch(self.table[ids,:512]),dim=-1,eps=1e-8)
            mask=~torch.eye(5,dtype=torch.bool)
            teacher=(t@t.T/0.1)[mask].reshape(5,4).softmax(-1).detach()
            student=(s@s.T/0.1)[mask].reshape(5,4).log_softmax(-1)
            reference=F.kl_div(student,teacher,reduction="batchmean")
            self.exact(model._compute_relation_loss(ids),reference)
            torch.manual_seed(51); out=model(self.feed())
            unique=torch.unique(torch.cat([self.feed()["history_items"].flatten(),self.feed()["item_id"].flatten()]))
            unique=unique[unique!=0]
            # Dropout consumes RNG before sampling; capture actual IDs, check
            # exact C0/C2 sampling separately with dropout disabled below.
            self.assertEqual(out["_relation_ids"].numel(),5)
            self.assertTrue(all(i in unique for i in out["_relation_ids"]))
            ranking=GeneralModel.loss(model,out)
            rel=model._compute_relation_loss(out["_relation_ids"])
            self.exact(model.loss(out),ranking+0.01*rel)
            model.zero_grad(); rel.backward()
            self.assertIsNone(model.item_encoder.complement_id_emb.weight.grad)
            self.assertTrue(model.item_encoder.semantic_branch[0].weight.grad.abs().sum()>0)
        for model in (clean,bridge):
            model.dropout.p=0
        bridge.backbone.proj.dropout_0.p=0
        torch.manual_seed(29); a=clean(self.feed())
        torch.manual_seed(29); b=bridge(self.feed())
        self.exact(a["_relation_ids"],b["_relation_ids"])

    def test_prompt_slots_empty_history_and_invalid_masks(self):
        for p in (0,3,5):
            backbone=PoMRecInterestBackbone(4,2,3,p,1,1.0)
            H=torch.zeros(2,3,4)
            mask=torch.tensor([[True,False,False],[False,False,False]])
            out=backbone(H,mask,torch.tensor([1,0]),True)
            self.assertEqual(backbone.prompt_pad.shape,(5-p,4))
            self.assertEqual(out["attention_maps"].shape,(2,2,8))
            self.assertTrue(torch.isfinite(out["interest_vectors"]).all())
            self.assertTrue((out["attention_maps"][:,:,1:3]==0).all())
            torch.testing.assert_close(out["attention_maps"].sum(-1),torch.ones(2,2))
        with self.assertRaises(ValueError):
            backbone(H,torch.tensor([[False,True,False],[False,False,False]]),torch.tensor([1,0]))
        for p in (-1,6):
            with self.assertRaises(ValueError):
                PoMRecInterestBackbone(4,2,3,p,1,1.0)


if __name__=="__main__":
    unittest.main(verbosity=2)
