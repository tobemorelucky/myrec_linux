# tools/export_pomrec_item_emb.py
import argparse
import pickle
import torch
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取tools/的绝对路径
sys.path.append(os.path.dirname(ROOT_DIR))             # 往上退一级，加入pom2.0/到搜索路径


from models.sequential.PoMRec import PoMRec  # 按你项目实际路径修改
from helpers.SeqReader import SeqReader         # 按你项目实际路径修改

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, required=True)
#     parser.add_argument("--ckpt", type=str, required=True, help="path to PoMRec checkpoint .pt")
#     parser.add_argument("--out", type=str, required=True, help="output pkl path")
#     args = parser.parse_args()
#
#     # 你项目里构建 corpus 的方式可能不同：这里给一个最常见的写法
#     dummy_args = type("obj", (), {})()
#     dummy_args.dataset = args.dataset
#     # 其他 reader 需要的参数你按自己项目补齐（例如 data_dir 等）
#     corpus = SeqReader(dummy_args).corpus  # 如果你项目不是这样构建，请按你自己的 reader 初始化方式改
#
#     model = PoMRec(dummy_args, corpus)
#     state = torch.load(args.ckpt, map_location="cpu")
#     model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()}, strict=False)
#     model.eval()
#
#     emb = model.interest_extractor.i_embeddings.weight.detach().cpu().numpy()  # (N1, emb)
#     # 保存不含padding 0行
#     emb_no_pad = emb[1:, :]
#     pickle.dump(emb_no_pad, open(args.out, "wb"))
#     print("saved:", args.out, emb_no_pad.shape)
#
# if __name__ == "__main__":
#     main()

# tools/export_pomrec_item_emb.py
import argparse
import pickle
import torch

def find_weight_key(state_dict):
    """
    兼容不同保存方式：
    - interest_extractor.i_embeddings.weight
    - model.interest_extractor.i_embeddings.weight
    - module.interest_extractor.i_embeddings.weight
    以及各种前缀
    """
    candidates = []
    for k, v in state_dict.items():
        if k.endswith("interest_extractor.i_embeddings.weight"):
            candidates.append(k)
    if not candidates:
        # 再放宽一点：只要包含这段路径
        for k, v in state_dict.items():
            if "interest_extractor.i_embeddings.weight" in k:
                candidates.append(k)
    if not candidates:
        raise KeyError(
            "Cannot find 'interest_extractor.i_embeddings.weight' in checkpoint. "
            "Please print state_dict.keys() to confirm the exact key."
        )
    # 如果有多个，优先最短的那个（通常前缀最少）
    candidates.sort(key=len)
    return candidates[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to PoMRec checkpoint .pt")
    parser.add_argument("--out", type=str, required=True, help="output pkl path")
    parser.add_argument("--keep_pad0", action="store_true",
                        help="If set, keep padding row 0. Default: drop row 0 and save (N, emb).")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")

    # 兼容 checkpoint 里可能包了一层 {'state_dict': ...} 或 {'model': ...}
    if isinstance(state, dict):
        for wrapper_key in ["state_dict", "model", "net"]:
            if wrapper_key in state and isinstance(state[wrapper_key], dict):
                state = state[wrapper_key]
                break

    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state)}")

    w_key = find_weight_key(state)
    W = state[w_key].detach().cpu().float().numpy()  # (N+1, emb) or (N, emb)

    # 与 LLMEmb 一致：通常保存不含 pad0 的 (N, emb)
    if not args.keep_pad0 and W.shape[0] >= 2:
        W_to_save = W[1:, :]
    else:
        W_to_save = W

    pickle.dump(W_to_save, open(args.out, "wb"))
    print(f"Found key: {w_key}")
    print(f"Saved: {args.out} shape={W_to_save.shape}")

if __name__ == "__main__":
    main()
