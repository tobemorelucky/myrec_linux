# -*- coding: UTF-8 -*-
"""
Analyze ASPCF / CHIR representations from a trained checkpoint.

Supports:
  - LLMMIRec
  - LLMMIRecASPCF
  - LLMMIRecCHIR
  - single-view / dual-view CHIR
  - prototype-global / prototype-specific calibration
  - optional prototype attention prior

Example (Dual View fused_split):
  python tools/analyze_llmmirec_aspcf.py \
    --checkpoint new_model/llmmirec_phase2d_dual/beauty/dual_fused_split/LLMMIRecCHIR_dual_fused_split_seed42.pt \
    --model_name LLMMIRecCHIR \
    --dataset beauty \
    --interest_query_mode prototype \
    --prototype_path ./data/beauty/handled/llmmi_proto32_sr512.pkl \
    --collab_calibration prototype \
    --routing_mode dual \
    --dual_view_source fused_split \
    --aspcf_gate_mode basic \
    --max_batches 50 \
    --output_dir new_log/llmmirec_phase2d_dual/beauty/diagnostics/fused_split
"""

import argparse
import json
import logging
import math
import os
import pickle
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn.functional as F


# =========================================================
# Args / helpers
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="LLMMIRec ASPCF/CHIR diagnostics"
    )

    p.add_argument("--checkpoint", type=str, required=True)

    p.add_argument(
        "--model_name",
        type=str,
        default="LLMMIRec",
        choices=[
            "LLMMIRec",
            "LLMMIRecASPCF",
            "LLMMIRecCHIR",
        ],
        help="Model class to instantiate",
    )

    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument(
        "--output_dir",
        type=str,
        default="./diagnostics_aspcf",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    # Prototype / CHIR configuration.
    # Prototype buffers are persistent=False and cannot always be
    # recovered from the state_dict, so explicit CLI overrides are
    # supported.
    p.add_argument(
        "--interest_query_mode",
        type=str,
        default="",
        choices=["", "learnable", "prototype"],
    )

    p.add_argument(
        "--prototype_path",
        type=str,
        default="",
    )

    p.add_argument(
        "--aspcf_gate_mode",
        type=str,
        default="",
        choices=["", "basic", "conflict"],
    )

    p.add_argument(
        "--collab_calibration",
        type=str,
        default="",
        choices=["", "global", "prototype"],
    )

    p.add_argument(
        "--routing_mode",
        type=str,
        default="",
        choices=["", "single", "dual"],
        help="Auto-detected from checkpoint when omitted",
    )

    p.add_argument(
        "--dual_view_source",
        type=str,
        default="",
        choices=["", "raw", "fused_split"],
        help="raw | fused_split",
    )

    p.add_argument(
        "--prototype_prior_strength",
        type=float,
        default=0.0,
        help="Only needed when diagnosing Phase2C prior checkpoints",
    )

    return p.parse_args()


def normalized_entropy(probs, dim=-1, eps=1e-8):
    """
    Normalized entropy in [0,1].
    """
    n = probs.shape[dim]

    h = -(
        probs * torch.log(probs + eps)
    ).sum(dim=dim)

    if n <= 1:
        return torch.zeros_like(h)

    return h / math.log(n)


def effective_rank_from_matrix(x, eps=1e-8):
    """
    Effective rank of [K,D] matrix.
    """
    s = torch.linalg.svdvals(
        x.float()
    )

    s = s[s > eps]

    if s.numel() == 0:
        return 0.0

    p = s / s.sum()

    entropy = -(
        p * torch.log(p + eps)
    ).sum()

    return float(
        torch.exp(entropy)
    )


def percentile(x, p):
    if isinstance(x, torch.Tensor):
        x = (
            x.detach()
            .cpu()
            .numpy()
        )

    return float(
        np.percentile(x, p)
    )


# =========================================================
# K-interest cosine helpers
# =========================================================

def per_user_k_cos_tensor(
    x,
    valid_lengths=None,
    eps=1e-8,
):
    """
    Compute K-way cosine similarity INSIDE EACH USER.

    Args:
        x:
            [B,K,D]
            or
            [B,K,L]

        valid_lengths:
            optional [B].
            If provided, the last dimension is interpreted as
            a sequence dimension and positions >= length are masked.

    Returns:
        [B], each element = mean pairwise cosine among this
        user's K vectors.

    Important:
        This does NOT flatten B*K together, so different users
        are never mixed when computing interest similarity.
    """

    x = x.float()

    if valid_lengths is not None:
        L = x.shape[-1]

        lens = (
            valid_lengths
            .long()
            .clamp(
                min=0,
                max=L,
            )
        )

        seq_mask = (
            torch.arange(
                L,
                device=x.device,
            )[None, :]
            <
            lens[:, None].to(x.device)
        ).float()

        x = (
            x
            *
            seq_mask[:, None, :]
        )

    B = x.shape[0]
    K = x.shape[1]

    if K <= 1:
        return torch.zeros(
            B,
            dtype=torch.float32,
            device=x.device,
        )

    x = x.reshape(
        B,
        K,
        -1,
    )

    x_n = F.normalize(
        x,
        dim=-1,
        eps=eps,
    )

    sim = torch.bmm(
        x_n,
        x_n.transpose(1, 2),
    )  # [B,K,K]

    upper_mask = torch.triu(
        torch.ones(
            K,
            K,
            dtype=torch.bool,
            device=x.device,
        ),
        diagonal=1,
    )

    pairwise = sim[:, upper_mask]

    return pairwise.mean(dim=-1)


def mean_per_user_k_cos(
    tensor_list,
    lengths_list=None,
    max_batches=None,
):
    """
    Average per-user K-way cosine across batches.
    """

    if not tensor_list:
        return 0.0

    n_batches = len(tensor_list)

    if max_batches is not None:
        n_batches = min(
            n_batches,
            max_batches,
        )

    values = []

    for batch_idx in range(n_batches):

        tensor = tensor_list[batch_idx]

        lengths = (
            lengths_list[batch_idx]
            if lengths_list is not None
            else None
        )

        per_user = per_user_k_cos_tensor(
            tensor,
            valid_lengths=lengths,
        )

        values.append(
            per_user.cpu()
        )

    if not values:
        return 0.0

    values = torch.cat(
        values,
        dim=0,
    )

    return float(
        values.mean()
    )


def project_tensor_list(
    tensor_list,
    linear_layer,
):
    """
    Apply a model Linear layer to CPU diagnostic tensors.

    tensor:
        [B,K,D_in]

    output:
        [B,K,D_out]
    """

    if not tensor_list:
        return []

    weight = (
        linear_layer.weight
        .detach()
        .cpu()
    )

    bias = (
        linear_layer.bias
        .detach()
        .cpu()
        if linear_layer.bias is not None
        else None
    )

    projected = []

    for tensor in tensor_list:

        projected.append(
            F.linear(
                tensor.float(),
                weight,
                bias,
            )
        )

    return projected


def masked_softmax_logits_list(
    logits_list,
    lengths_list,
):
    """
    DualViewInterestExtractor returns semantic/collaborative logits
    BEFORE the padding mask.

    For diagnostic softmax, padding therefore must be masked again.

    Input:
        logits: [B,K,L]
        lengths: [B]

    Output:
        attention: [B,K,L]
    """

    outputs = []

    for logits, lengths in zip(
        logits_list,
        lengths_list,
    ):

        scores = (
            logits
            .float()
            .clone()
        )

        L = scores.shape[-1]

        lens = (
            lengths
            .long()
            .clamp(
                min=0,
                max=L,
            )
        )

        valid = (
            torch.arange(L)[None, :]
            <
            lens[:, None]
        )

        scores = scores.masked_fill(
            ~valid[:, None, :],
            float("-inf"),
        )

        attn = F.softmax(
            scores,
            dim=-1,
        )

        attn = torch.nan_to_num(
            attn,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        outputs.append(attn)

    return outputs


# =========================================================
# Model / checkpoint reconstruction
# =========================================================

def select_model_class(model_name):

    if model_name == "LLMMIRecASPCF":
        from models.sequential.LLMMIRecASPCF import (
            LLMMIRecASPCF,
        )

        return LLMMIRecASPCF

    if model_name == "LLMMIRecCHIR":
        from models.sequential.LLMMIRecCHIR import (
            LLMMIRecCHIR,
        )

        return LLMMIRecCHIR

    from models.sequential.LLMMIRec import (
        LLMMIRec,
    )

    return LLMMIRec


def infer_item_encoder_mode(state):

    has_adapter = (
        "item_encoder.adapter.0.weight"
        in state
    )

    has_log_gamma = (
        "item_encoder.log_gamma"
        in state
    )

    has_gamma_buffer = (
        "item_encoder.gamma"
        in state
    )

    has_semantic = (
        "item_encoder.semantic_branch.0.weight"
        in state
    )

    if has_semantic:
        return "aspcf"

    if (
        has_adapter
        and
        (
            has_log_gamma
            or has_gamma_buffer
        )
    ):
        return "residual"

    if has_adapter:
        return "llm_replace"

    return "id"


def infer_k(state):
    """
    Use aggregator output dimension.
    This works for both single and dual routing.
    """

    key = "aggregator.mlp.2.weight"

    if key in state:
        return int(
            state[key].shape[0]
        )

    # Historical fallback.
    if "extractor.query" in state:
        return int(
            state[
                "extractor.query"
            ].shape[0]
        )

    raise KeyError(
        "Cannot infer K: neither "
        "aggregator.mlp.2.weight nor "
        "extractor.query exists."
    )


def infer_gate_mode(
    state,
    semantic_dim,
    complement_dim,
):
    """
    basic:
        [s,c]
        input = semantic_dim + complement_dim

    conflict:
        [s,c,abs(s-c),s*c]
        input = 2 * (semantic_dim + complement_dim)
    """

    gate_key = (
        "item_encoder.gate.0.weight"
    )

    if gate_key not in state:
        return "basic"

    gate_input_dim = int(
        state[gate_key].shape[1]
    )

    basic_dim = (
        semantic_dim
        +
        complement_dim
    )

    conflict_dim = (
        2
        *
        basic_dim
    )

    if gate_input_dim == basic_dim:
        return "basic"

    if gate_input_dim == conflict_dim:
        return "conflict"

    raise ValueError(
        "Cannot infer ASPCF gate mode: "
        f"gate input={gate_input_dim}, "
        f"basic expected={basic_dim}, "
        f"conflict expected={conflict_dim}"
    )


def build_model_args(
    args,
    state,
    mode,
):

    class DummyArgs:
        pass

    ma = DummyArgs()

    # -------------------------
    # Base framework arguments
    # -------------------------

    ma.device = torch.device(
        args.device
    )

    ma.model_path = (
        args.checkpoint
    )

    ma.buffer = 1
    ma.history_max = 20
    ma.num_neg = 1
    ma.test_all = 0

    # -------------------------
    # Core dimensions
    # -------------------------

    ma.emb_size = int(
        state[
            "position_emb.weight"
        ].shape[1]
    )

    ma.K = infer_k(state)

    # -------------------------
    # Detect single / dual
    # -------------------------

    is_dual_checkpoint = (
        "extractor.Wq_sem.weight"
        in state
    )

    detected_routing = (
        "dual"
        if is_dual_checkpoint
        else "single"
    )

    if (
        args.routing_mode
        and
        args.routing_mode
        != detected_routing
    ):
        raise ValueError(
            f"--routing_mode={args.routing_mode} "
            "conflicts with checkpoint. "
            f"Detected routing={detected_routing}"
        )

    ma.routing_mode = (
        detected_routing
    )

    if is_dual_checkpoint:

        ma.attn_size = int(
            state[
                "extractor.Wq_sem.weight"
            ].shape[0]
        )

        ma.routing_gate_hidden = int(
            state[
                "extractor.routing_gate.0.weight"
            ].shape[0]
        )

    else:

        ma.attn_size = int(
            state[
                "extractor.Wq.weight"
            ].shape[0]
        )

        ma.routing_gate_hidden = 32

    # -------------------------
    # Item encoder
    # -------------------------

    ma.item_encoder = mode

    ma.llm_emb_path = (
        f"./data/{args.dataset}/handled/"
        "llm_table_pca1536.pkl"
        if mode != "id"
        else ""
    )

    if (
        "item_encoder.adapter.0.weight"
        in state
    ):

        ma.adapter_hidden = int(
            state[
                "item_encoder.adapter.0.weight"
            ].shape[0]
        )

    else:
        ma.adapter_hidden = 256

    ma.adapter_activation = "gelu"

    ma.adapter_use_ln = int(
        any(
            key.startswith(
                "item_encoder.adapter.3"
            )
            for key in state.keys()
        )
    )

    ma.gamma_init = 0.1

    ma.gamma_trainable = int(
        "item_encoder.log_gamma"
        in state
    )

    ma.dropout = 0.1

    # -------------------------
    # ASPCF dimensions
    # -------------------------

    has_semantic = (
        "item_encoder.semantic_branch.0.weight"
        in state
    )

    if has_semantic:

        ma.semantic_rank = int(
            state[
                "item_encoder.semantic_branch.0.weight"
            ].shape[1]
        )

        ma.semantic_hidden = int(
            state[
                "item_encoder.semantic_branch.0.weight"
            ].shape[0]
        )

        ma.semantic_dim = int(
            state[
                "item_encoder.semantic_branch.2.weight"
            ].shape[0]
        )

        ma.tail_hidden = int(
            state[
                "item_encoder.complement_tail.0.weight"
            ].shape[0]
        )

        ma.complement_hidden = int(
            state[
                "item_encoder.complement_mlp.0.weight"
            ].shape[0]
        )

        ma.complement_dim = int(
            state[
                "item_encoder.complement_mlp.2.weight"
            ].shape[0]
        )

        ma.gate_hidden = int(
            state[
                "item_encoder.gate.0.weight"
            ].shape[0]
        )

    else:

        ma.semantic_rank = 512
        ma.semantic_hidden = 128
        ma.semantic_dim = 32

        ma.tail_hidden = 64
        ma.complement_hidden = 64
        ma.complement_dim = 32

        ma.gate_hidden = 64

    # -------------------------
    # Gate mode
    # -------------------------

    if has_semantic:

        inferred_gate = infer_gate_mode(
            state,
            ma.semantic_dim,
            ma.complement_dim,
        )

    else:

        inferred_gate = "basic"

    if (
        args.aspcf_gate_mode
        and
        args.aspcf_gate_mode
        != inferred_gate
    ):

        raise ValueError(
            f"--aspcf_gate_mode="
            f"{args.aspcf_gate_mode} "
            "conflicts with checkpoint. "
            f"Detected gate={inferred_gate}"
        )

    ma.aspcf_gate_mode = (
        args.aspcf_gate_mode
        or inferred_gate
    )

    # -------------------------
    # Relation loss
    # -------------------------

    # model.eval() does not compute relation loss,
    # but these attributes are still needed.
    ma.lambda_relation = 0.0
    ma.relation_sample_size = 128
    ma.relation_teacher_temp = 0.1
    ma.relation_student_temp = 0.1

    # -------------------------
    # Prototype / CHIR
    # -------------------------

    if args.interest_query_mode:

        ma.interest_query_mode = (
            args.interest_query_mode
        )

    elif args.prototype_path:

        ma.interest_query_mode = (
            "prototype"
        )

    else:

        ma.interest_query_mode = (
            "learnable"
        )

    ma.prototype_path = (
        args.prototype_path
        or ""
    )

    if args.collab_calibration:

        ma.collab_calibration = (
            args.collab_calibration
        )

    elif (
        ma.routing_mode == "dual"
        and
        ma.interest_query_mode
        == "prototype"
    ):

        # Current Phase2D setup.
        ma.collab_calibration = (
            "prototype"
        )

    else:

        ma.collab_calibration = (
            "global"
        )

    ma.prototype_prior_strength = float(
        args.prototype_prior_strength
    )

    ma.dual_view_source = (
        args.dual_view_source
        or "raw"
    )

    # -------------------------
    # Validation
    # -------------------------

    if (
        ma.interest_query_mode
        == "prototype"
        and
        not ma.prototype_path
    ):

        raise ValueError(
            "Prototype mode requires "
            "--prototype_path. "
            "Prototype centers/assignments "
            "are persistent=False and are not "
            "stored in the checkpoint."
        )

    if (
        ma.routing_mode == "dual"
        and
        ma.interest_query_mode
        != "prototype"
    ):

        raise ValueError(
            "Current LLMMIRecCHIR "
            "dual routing requires "
            "--interest_query_mode prototype."
        )

    logging.info(
        "Reconstructed config: "
        f"K={ma.K}, "
        f"emb={ma.emb_size}, "
        f"attn={ma.attn_size}, "
        f"routing={ma.routing_mode}, "
        f"query={ma.interest_query_mode}, "
        f"calibration={ma.collab_calibration}, "
        f"gate={ma.aspcf_gate_mode}, "
        f"dual_view_source={ma.dual_view_source}, "
        f"prior_strength="
        f"{ma.prototype_prior_strength}"
    )

    return ma


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load corpus / checkpoint
    # -----------------------------------------------------

    if not os.path.exists(
        args.checkpoint
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    corpus_path = (
        f"./data/{args.dataset}/"
        "SeqReader.pkl"
    )

    if not os.path.exists(
        corpus_path
    ):
        raise FileNotFoundError(
            f"Corpus not found: "
            f"{corpus_path}"
        )

    with open(
        corpus_path,
        "rb",
    ) as f:

        corpus = pickle.load(f)

    state = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    if not isinstance(
        state,
        dict,
    ):
        raise TypeError(
            "Expected state_dict-like dict, "
            f"got {type(state)}"
        )

    mode = infer_item_encoder_mode(
        state
    )

    is_dual = (
        "extractor.Wq_sem.weight"
        in state
    )

    logging.info(
        f"Model: {args.model_name}, "
        f"item_encoder={mode}, "
        f"routing="
        f"{'dual' if is_dual else 'single'}"
    )

    # -----------------------------------------------------
    # Reconstruct model
    # -----------------------------------------------------

    ModelClass = select_model_class(
        args.model_name
    )

    ma = build_model_args(
        args,
        state,
        mode,
    )

    model = ModelClass(
        ma,
        corpus,
    ).to(
        args.device
    )

    # Diagnostics should use an architecture that exactly
    # matches the checkpoint. Fail loudly otherwise.
    try:

        model.load_state_dict(
            state,
            strict=True,
        )

    except RuntimeError as e:

        raise RuntimeError(
            "Checkpoint architecture does not "
            "match reconstructed model args. "
            "Check --model_name, "
            "--interest_query_mode, "
            "--prototype_path, "
            "--routing_mode, "
            "--dual_view_source and "
            "--aspcf_gate_mode."
        ) from e

    model.eval()

    logging.info(
        f"Loaded: "
        f"#params={model.count_variables()}"
    )

    # -----------------------------------------------------
    # Test dataset
    # -----------------------------------------------------

    dataset = ModelClass.Dataset(
        model,
        corpus,
        "test",
    )

    dataset.prepare()

    from torch.utils.data import (
        DataLoader,
    )

    dl = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.collate_batch,
    )

    # -----------------------------------------------------
    # Accumulators
    # -----------------------------------------------------

    all_iv = []
    all_attn = []
    all_w = []
    all_lengths = []

    all_alpha_sem = []
    all_alpha_comp = []

    all_h_semantic = []
    all_h_complement = []

    all_proto_mass = []
    all_proto_ids = []

    all_proto_hist_w = []
    all_proto_collab_ctx = []

    all_query_seeds = []

    all_logits_before = []
    all_attn_prior = []

    # Dual-view-specific
    all_rho = []

    all_sem_logits = []
    all_comp_logits = []

    all_sem_query = []
    all_collab_query = []

    batch_count = 0

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    with torch.inference_mode():

        for batch in dl:

            batch = {
                key: (
                    value.to(args.device)
                    if isinstance(
                        value,
                        torch.Tensor,
                    )
                    else value
                )
                for key, value
                in batch.items()
            }

            out = model(
                batch,
                return_intermediate=True,
            )

            all_iv.append(
                out[
                    "interest_vectors"
                ].detach().cpu()
            )

            all_attn.append(
                out[
                    "attention_maps"
                ].detach().cpu()
            )

            all_w.append(
                out[
                    "interest_weights"
                ].detach().cpu()
            )

            all_lengths.append(
                batch[
                    "lengths"
                ].detach().cpu()
            )

            # -----------------------------
            # ASPCF
            # -----------------------------

            if (
                mode == "aspcf"
                and
                "history_alpha_sem"
                in out
            ):

                all_alpha_sem.append(
                    out[
                        "history_alpha_sem"
                    ].detach().cpu()
                )

                all_alpha_comp.append(
                    out[
                        "history_alpha_comp"
                    ].detach().cpu()
                )

                all_h_semantic.append(
                    out[
                        "history_semantic"
                    ].detach().cpu()
                )

                all_h_complement.append(
                    out[
                        "history_complement"
                    ].detach().cpu()
                )

            # -----------------------------
            # Prototype
            # -----------------------------

            if (
                "prototype_mass"
                in out
            ):

                all_proto_mass.append(
                    out[
                        "prototype_mass"
                    ].detach().cpu()
                )

                all_proto_ids.append(
                    out[
                        "selected_prototype_ids"
                    ].detach().cpu()
                )

            if (
                "prototype_history_weights"
                in out
            ):

                all_proto_hist_w.append(
                    out[
                        "prototype_history_weights"
                    ].detach().cpu()
                )

                all_proto_collab_ctx.append(
                    out[
                        "prototype_collab_context"
                    ].detach().cpu()
                )

            if (
                "query_seeds"
                in out
            ):

                all_query_seeds.append(
                    out[
                        "query_seeds"
                    ].detach().cpu()
                )

            # -----------------------------
            # Phase2C prior
            # -----------------------------

            if (
                "attention_logits_before_prior"
                in out
            ):

                all_logits_before.append(
                    out[
                        "attention_logits_before_prior"
                    ].detach().cpu()
                )

                all_attn_prior.append(
                    out[
                        "attention_prior"
                    ].detach().cpu()
                )

            # -----------------------------
            # Dual view
            # -----------------------------

            if (
                "routing_rho"
                in out
            ):

                all_rho.append(
                    out[
                        "routing_rho"
                    ].detach().cpu()
                )

                all_sem_logits.append(
                    out[
                        "semantic_attention_logits"
                    ].detach().cpu()
                )

                all_comp_logits.append(
                    out[
                        "collaborative_attention_logits"
                    ].detach().cpu()
                )

                # IMPORTANT:
                #
                # semantic_query:
                #   [B,K,32]
                #
                # collaborative_query:
                #   [B,K,32]
                #
                # query_seeds:
                #   concat -> [B,K,64]
                #
                # Wq_sem/Wq_comp both expect 32 dimensions.
                #
                # Therefore Wq_sem must NEVER be applied
                # directly to query_seeds.
                all_sem_query.append(
                    out[
                        "semantic_query"
                    ].detach().cpu()
                )

                all_collab_query.append(
                    out[
                        "collaborative_query"
                    ].detach().cpu()
                )

            batch_count += 1

            if (
                args.max_batches > 0
                and
                batch_count
                >= args.max_batches
            ):
                break

    if not all_iv:

        raise RuntimeError(
            "No test batches were processed."
        )

    # -----------------------------------------------------
    # Concatenate basic outputs
    # -----------------------------------------------------

    iv = torch.cat(
        all_iv,
        dim=0,
    )

    w = torch.cat(
        all_w,
        dim=0,
    )

    lengths = torch.cat(
        all_lengths,
        dim=0,
    )

    N, K, D = iv.shape

    logging.info(
        f"Samples: {N}, "
        f"K={K}, "
        f"D={D}"
    )

    # =====================================================
    # Basic interest diagnostics
    # =====================================================

    pairwise_values = (
        per_user_k_cos_tensor(iv)
    )

    mean_pairwise = float(
        pairwise_values.mean()
    )

    # Maximum inter-interest cosine.
    iv_n = F.normalize(
        iv.float(),
        dim=-1,
        eps=1e-8,
    )

    sim_mat = torch.bmm(
        iv_n,
        iv_n.transpose(1, 2),
    )

    if K > 1:

        diagonal = torch.eye(
            K,
            dtype=torch.bool,
        ).unsqueeze(0)

        sim_no_diag = (
            sim_mat.masked_fill(
                diagonal,
                float("-inf"),
            )
        )

        mean_max_inter = float(
            sim_no_diag.amax(
                dim=(-1, -2)
            ).mean()
        )

    else:

        mean_max_inter = 0.0

    # Effective rank.
    eff_ranks = []

    for i in range(
        min(N, 1000)
    ):

        eff_ranks.append(
            effective_rank_from_matrix(
                iv[i]
            )
        )

    mean_eff_rank = (
        float(np.mean(eff_ranks))
        if eff_ranks
        else 0.0
    )

    # Attention entropy.
    attn_entropies = []

    for attn_batch, lens in zip(
        all_attn,
        all_lengths,
    ):

        B = attn_batch.shape[0]

        for i in range(B):

            valid_len = int(
                lens[i].item()
            )

            if valid_len <= 0:
                continue

            probs = (
                attn_batch[
                    i,
                    :,
                    :valid_len,
                ].float()
            )  # [K,L]

            if valid_len <= 1:

                entropy = torch.zeros(
                    K
                )

            else:

                entropy = -(
                    probs
                    *
                    torch.log(
                        probs + 1e-8
                    )
                ).sum(
                    dim=-1
                )

                entropy = (
                    entropy
                    /
                    math.log(valid_len)
                )

            attn_entropies.extend(
                entropy.tolist()
            )

    mean_attn_entropy = (
        float(
            np.mean(
                attn_entropies
            )
        )
        if attn_entropies
        else 0.0
    )

    # Interest aggregation entropy.
    weight_entropy = (
        normalized_entropy(
            w.float(),
            dim=-1,
        )
    )

    mean_weight_entropy = float(
        weight_entropy.mean()
    )

    mean_max_interest_weight = float(
        w.max(
            dim=-1
        ).values.mean()
    )

    # Final attention K similarity.
    mean_query_attn_cos = (
        mean_per_user_k_cos(
            all_attn,
            lengths_list=all_lengths,
        )
    )

    stats = {

        "checkpoint":
            args.checkpoint,

        "model_name":
            args.model_name,

        "mode":
            mode,

        "routing_mode":
            ma.routing_mode,

        "dual_view_source":
            ma.dual_view_source,

        "interest_query_mode":
            ma.interest_query_mode,

        "collab_calibration":
            ma.collab_calibration,

        "num_samples":
            N,

        "K":
            K,

        "D":
            D,

        "mean_pairwise_cos":
            round(
                mean_pairwise,
                6,
            ),

        "mean_max_inter_sim":
            round(
                mean_max_inter,
                6,
            ),

        "mean_effective_rank":
            round(
                mean_eff_rank,
                3,
            ),

        "mean_attn_entropy":
            round(
                mean_attn_entropy,
                6,
            ),

        "mean_weight_entropy":
            round(
                mean_weight_entropy,
                6,
            ),

        "mean_max_interest_weight":
            round(
                mean_max_interest_weight,
                6,
            ),

        "mean_query_attn_cos":
            round(
                mean_query_attn_cos,
                6,
            ),
    }

    # =====================================================
    # ASPCF diagnostics
    # =====================================================

    if (
        mode == "aspcf"
        and
        all_alpha_sem
    ):

        semantic_alpha_values = []
        complement_alpha_values = []

        semantic_complement_cos = []

        for (
            alpha_sem,
            alpha_comp,
            history_sem,
            history_comp,
            lens,
        ) in zip(

            all_alpha_sem,
            all_alpha_comp,
            all_h_semantic,
            all_h_complement,
            all_lengths,
        ):

            L = alpha_sem.shape[1]

            valid_mask = (
                torch.arange(L)[None, :]
                <
                lens
                .long()
                .clamp(
                    min=0,
                    max=L,
                )[:, None]
            )

            semantic_alpha_values.append(
                alpha_sem[
                    valid_mask
                ].float()
            )

            complement_alpha_values.append(
                alpha_comp[
                    valid_mask
                ].float()
            )

            # Only compare directly if branch dimensions match.
            if (
                history_sem.shape[-1]
                ==
                history_comp.shape[-1]
            ):

                s = (
                    history_sem[
                        valid_mask
                    ].float()
                )

                c = (
                    history_comp[
                        valid_mask
                    ].float()
                )

                if s.numel() > 0:

                    s_n = F.normalize(
                        s,
                        dim=-1,
                        eps=1e-8,
                    )

                    c_n = F.normalize(
                        c,
                        dim=-1,
                        eps=1e-8,
                    )

                    semantic_complement_cos.append(
                        (
                            s_n
                            *
                            c_n
                        ).sum(
                            dim=-1
                        )
                    )

        if semantic_alpha_values:

            alpha_sem_values = torch.cat(
                semantic_alpha_values
            )

            alpha_comp_values = torch.cat(
                complement_alpha_values
            )

            for (
                name,
                values,
            ) in [

                (
                    "alpha_sem",
                    alpha_sem_values,
                ),

                (
                    "alpha_comp",
                    alpha_comp_values,
                ),
            ]:

                stats[
                    f"{name}_mean"
                ] = round(
                    float(
                        values.mean()
                    ),
                    6,
                )

                stats[
                    f"{name}_std"
                ] = round(
                    float(
                        values.std()
                    ),
                    6,
                )

                stats[
                    f"{name}_min"
                ] = round(
                    float(
                        values.min()
                    ),
                    6,
                )

                stats[
                    f"{name}_max"
                ] = round(
                    float(
                        values.max()
                    ),
                    6,
                )

                stats[
                    f"{name}_p10"
                ] = round(
                    percentile(
                        values,
                        10,
                    ),
                    6,
                )

                stats[
                    f"{name}_p50"
                ] = round(
                    percentile(
                        values,
                        50,
                    ),
                    6,
                )

                stats[
                    f"{name}_p90"
                ] = round(
                    percentile(
                        values,
                        90,
                    ),
                    6,
                )

        if semantic_complement_cos:

            sc_cos = torch.cat(
                semantic_complement_cos
            )

            stats[
                "semantic_complement_cos_mean"
            ] = round(
                float(
                    sc_cos.mean()
                ),
                6,
            )

            stats[
                "semantic_complement_cos_std"
            ] = round(
                float(
                    sc_cos.std()
                ),
                6,
            )

    # =====================================================
    # Prototype diagnostics
    # =====================================================

    if all_proto_mass:

        proto_mass = torch.cat(
            all_proto_mass,
            dim=0,
        ).float()

        proto_ids = torch.cat(
            all_proto_ids,
            dim=0,
        ).long()

        proto_num = (
            proto_mass.shape[1]
        )

        # -----------------------------
        # Mean assignment mass
        # -----------------------------

        proto_usage = (
            proto_mass.mean(
                dim=0
            )
        )

        for p in range(
            proto_num
        ):

            stats[
                f"proto{p}_usage"
            ] = round(
                float(
                    proto_usage[p]
                ),
                6,
            )

        # -----------------------------
        # Selection frequency
        # -----------------------------

        selected_flat = (
            proto_ids.reshape(-1)
        )

        for p in range(
            proto_num
        ):

            freq = (
                selected_flat == p
            ).float().mean()

            stats[
                f"proto{p}_select_freq"
            ] = round(
                float(freq),
                6,
            )

        # -----------------------------
        # Duplicate rate
        # -----------------------------

        duplicate_users = 0

        for row in proto_ids:

            if (
                torch.unique(
                    row
                ).numel()
                <
                K
            ):

                duplicate_users += 1

        stats[
            "proto_dup_rate"
        ] = round(
            duplicate_users
            /
            max(
                proto_ids.shape[0],
                1,
            ),
            6,
        )

        # -----------------------------
        # Prototype mass entropy
        # -----------------------------

        pm = (
            proto_mass
            .clamp_min(1e-8)
        )

        pm = (
            pm
            /
            pm.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        proto_entropy = (
            normalized_entropy(
                pm,
                dim=-1,
            )
        )

        stats[
            "proto_mass_entropy_mean"
        ] = round(
            float(
                proto_entropy.mean()
            ),
            6,
        )

        stats[
            "proto_mass_entropy_std"
        ] = round(
            float(
                proto_entropy.std()
            ),
            6,
        )

        # -----------------------------
        # Selected prototype similarity
        # -----------------------------

        if (
            hasattr(
                model,
                "proto_centers",
            )
            and
            proto_ids.numel() > 0
        ):

            centers = (
                model.proto_centers
                .detach()
                .cpu()
                .float()
            )

            n_eval = min(
                proto_ids.shape[0],
                1000,
            )

            selected_centers = (
                centers[
                    proto_ids[:n_eval]
                ]
            )  # [N,K,R]

            selected_cos = (
                per_user_k_cos_tensor(
                    selected_centers
                )
            )

            stats[
                "proto_selected_pairwise_cos"
            ] = round(
                float(
                    selected_cos.mean()
                ),
                6,
            )

    # =====================================================
    # Prototype-specific historical routing
    # =====================================================

    if all_proto_hist_w:

        # -----------------------------
        # History weight entropy
        # -----------------------------

        entropy_values = []

        processed_users = 0
        max_users = 1000

        for (
            hist_weights,
            lens,
        ) in zip(
            all_proto_hist_w,
            all_lengths,
        ):

            B, L, K_hw = (
                hist_weights.shape
            )

            for i in range(B):

                if (
                    processed_users
                    >= max_users
                ):
                    break

                valid_len = int(
                    lens[i].item()
                )

                if valid_len <= 0:

                    processed_users += 1
                    continue

                probs = (
                    hist_weights[
                        i,
                        :valid_len,
                        :,
                    ]
                    .t()
                    .float()
                )  # [K,L]

                probs = (
                    probs
                    /
                    probs.sum(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-8)
                )

                if valid_len <= 1:

                    entropy = (
                        torch.zeros(
                            K_hw
                        )
                    )

                else:

                    entropy = -(
                        probs
                        *
                        torch.log(
                            probs + 1e-8
                        )
                    ).sum(
                        dim=-1
                    )

                    entropy = (
                        entropy
                        /
                        math.log(valid_len)
                    )

                entropy_values.extend(
                    entropy.tolist()
                )

                processed_users += 1

            if (
                processed_users
                >= max_users
            ):
                break

        stats[
            "proto_hist_weight_entropy_mean"
        ] = round(
            float(
                np.mean(
                    entropy_values
                )
            )
            if entropy_values
            else 0.0,
            6,
        )

        # -----------------------------
        # K routing cosine
        # -----------------------------

        # [B,L,K] -> [B,K,L]
        history_weight_k = [

            tensor
            .permute(
                0,
                2,
                1,
            )
            .contiguous()

            for tensor
            in all_proto_hist_w
        ]

        stats[
            "proto_hist_weight_cos_mean"
        ] = round(
            mean_per_user_k_cos(
                history_weight_k,
                lengths_list=all_lengths,
            ),
            6,
        )

    # Collaborative context similarity.
    if all_proto_collab_ctx:

        stats[
            "proto_collab_context_inter_k_cos"
        ] = round(
            mean_per_user_k_cos(
                all_proto_collab_ctx
            ),
            6,
        )

    # =====================================================
    # Query diagnostics
    # =====================================================

    # Combined query seed [semantic_query ; collaborative_query].
    if all_query_seeds:

        stats[
            "query_seed_inter_k_cos"
        ] = round(
            mean_per_user_k_cos(
                all_query_seeds
            ),
            6,
        )

    # -----------------------------------------------------
    # Dual-view:
    #
    # semantic_query:      32d -> Wq_sem
    # collaborative_query: 32d -> Wq_comp
    #
    # DO NOT:
    # 64d query_seed -> Wq_sem
    # -----------------------------------------------------

    if is_dual:

        if all_sem_query:

            stats[
                "semantic_query_inter_k_cos"
            ] = round(
                mean_per_user_k_cos(
                    all_sem_query
                ),
                6,
            )

            semantic_projected = (
                project_tensor_list(
                    all_sem_query,
                    model.extractor.Wq_sem,
                )
            )

            stats[
                "Wq_sem_query_inter_k_cos"
            ] = round(
                mean_per_user_k_cos(
                    semantic_projected
                ),
                6,
            )

        if all_collab_query:

            stats[
                "collaborative_query_inter_k_cos"
            ] = round(
                mean_per_user_k_cos(
                    all_collab_query
                ),
                6,
            )

            collaborative_projected = (
                project_tensor_list(
                    all_collab_query,
                    model.extractor.Wq_comp,
                )
            )

            stats[
                "Wq_comp_query_inter_k_cos"
            ] = round(
                mean_per_user_k_cos(
                    collaborative_projected
                ),
                6,
            )

    # Single-view:
    # query_seed is 64d and extractor.Wq expects 64d.
    else:

        if (
            all_query_seeds
            and
            hasattr(
                model.extractor,
                "Wq",
            )
        ):

            projected_query_seeds = (
                project_tensor_list(
                    all_query_seeds,
                    model.extractor.Wq,
                )
            )

            stats[
                "Wq_query_seed_inter_k_cos"
            ] = round(
                mean_per_user_k_cos(
                    projected_query_seeds
                ),
                6,
            )

    # =====================================================
    # Phase2C prior diagnostics
    # =====================================================

    if all_attn_prior:

        stats[
            "attention_prior_inter_k_cos"
        ] = round(
            mean_per_user_k_cos(
                all_attn_prior,
                lengths_list=all_lengths,
            ),
            6,
        )

    if all_logits_before:

        stats[
            "logits_before_prior_inter_k_cos"
        ] = round(
            mean_per_user_k_cos(
                all_logits_before,
                lengths_list=all_lengths,
            ),
            6,
        )

    # =====================================================
    # Dual-view routing diagnostics
    # =====================================================

    if all_rho:

        # -----------------------------
        # Routing rho
        # -----------------------------

        rho = torch.cat(
            all_rho,
            dim=0,
        ).float()

        stats[
            "routing_rho_mean"
        ] = round(
            float(
                rho.mean()
            ),
            6,
        )

        stats[
            "routing_rho_std"
        ] = round(
            float(
                rho.std()
            ),
            6,
        )

        stats[
            "routing_rho_p10"
        ] = round(
            percentile(
                rho,
                10,
            ),
            6,
        )

        stats[
            "routing_rho_p50"
        ] = round(
            percentile(
                rho,
                50,
            ),
            6,
        )

        stats[
            "routing_rho_p90"
        ] = round(
            percentile(
                rho,
                90,
            ),
            6,
        )

        # Variance among K interests for each user.
        rho_var = rho.var(
            dim=-1,
            unbiased=False,
        )

        stats[
            "routing_rho_per_sample_var"
        ] = round(
            float(
                rho_var.mean()
            ),
            6,
        )

        rho_range = (
            rho.max(
                dim=-1
            ).values
            -
            rho.min(
                dim=-1
            ).values
        )

        stats[
            "routing_rho_per_sample_range_mean"
        ] = round(
            float(
                rho_range.mean()
            ),
            6,
        )

        # -----------------------------
        # Branch logits cosine
        # -----------------------------
        #
        # semantic_attention_logits and
        # collaborative_attention_logits are returned
        # BEFORE the padding mask.
        #
        # Padding is therefore explicitly masked to zero
        # when computing cosine.

        stats[
            "semantic_attn_logits_k_cos"
        ] = round(
            mean_per_user_k_cos(
                all_sem_logits,
                lengths_list=all_lengths,
            ),
            6,
        )

        stats[
            "collab_attn_logits_k_cos"
        ] = round(
            mean_per_user_k_cos(
                all_comp_logits,
                lengths_list=all_lengths,
            ),
            6,
        )

        # -----------------------------
        # Branch attention cosine
        # -----------------------------

        semantic_attention = (
            masked_softmax_logits_list(
                all_sem_logits,
                all_lengths,
            )
        )

        collaborative_attention = (
            masked_softmax_logits_list(
                all_comp_logits,
                all_lengths,
            )
        )

        stats[
            "semantic_attn_softmax_k_cos"
        ] = round(
            mean_per_user_k_cos(
                semantic_attention,
                lengths_list=all_lengths,
            ),
            6,
        )

        stats[
            "collab_attn_softmax_k_cos"
        ] = round(
            mean_per_user_k_cos(
                collaborative_attention,
                lengths_list=all_lengths,
            ),
            6,
        )

        # -----------------------------
        # Final fused attention cosine
        # -----------------------------

        stats[
            "final_attn_k_cos"
        ] = round(
            mean_per_user_k_cos(
                all_attn,
                lengths_list=all_lengths,
            ),
            6,
        )

    # =====================================================
    # Save
    # =====================================================

    json_path = os.path.join(
        args.output_dir,
        "stats.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logging.info(
        f"Saved: {json_path}"
    )

    tsv_path = os.path.join(
        args.output_dir,
        "stats.tsv",
    )

    with open(
        tsv_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "key\tvalue\n"
        )

        for key, value in stats.items():

            f.write(
                f"{key}\t{value}\n"
            )

    logging.info(
        f"Saved: {tsv_path}"
    )

    print(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    main()