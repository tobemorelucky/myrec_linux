# tools/convert_llmemb_jsonl_to_pkl_table.py
import argparse
import json
import pickle
import numpy as np
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="LLMEmb output jsonlines (each line has item_id, hidden_states)")
    parser.add_argument("--item2id", type=str, required=True, help="PoMRec item2id.json (asin->id)")
    parser.add_argument("--out", type=str, required=True, help="output pkl path for table (N+1,d)")
    parser.add_argument("--pca_dim", type=int, default=0, help="if >0, also save PCA version")
    parser.add_argument("--out_pca", type=str, default="", help="output pkl for PCA table")
    args = parser.parse_args()

    item2id = json.load(open(args.item2id, "r", encoding="utf-8"))
    N = max(item2id.values()) if isinstance(next(iter(item2id.values())), int) else max(int(v) for v in item2id.values())
    # 标准：table size = N+1
    table = None
    filled = np.zeros((N + 1,), dtype=np.int8)

    # 先扫一遍确定 d
    lines = open(args.jsonl, "r", encoding="utf-8").read().splitlines()
    if len(lines) == 0:
        raise RuntimeError("empty jsonl")
    first = json.loads(lines[0])
    d = len(first["hidden_states"])
    table = np.zeros((N + 1, d), dtype=np.float32)

    for line in lines:
        obj = json.loads(line)
        item_id = int(obj["item_id"])
        vec = np.asarray(obj["hidden_states"], dtype=np.float32)
        if vec.shape[0] != d:
            raise ValueError(f"dim mismatch for item_id={item_id}: {vec.shape[0]} vs {d}")
        if item_id < 0 or item_id > N:
            # 超界直接报错最安全（说明映射不一致）
            raise ValueError(f"item_id out of range: {item_id}, expected 1..{N}")
        if item_id == 0:
            continue
        table[item_id] = vec
        filled[item_id] = 1

    missing = np.where(filled[1:] == 0)[0] + 1
    if len(missing) > 0:
        raise RuntimeError(f"Missing item embeddings count={len(missing)}. First 20 missing: {missing[:20].tolist()}")

    pickle.dump(table, open(args.out, "wb"))
    print("Saved table:", args.out, table.shape)

    if args.pca_dim and args.pca_dim > 0:
        pca = PCA(n_components=args.pca_dim)
        pca_emb = pca.fit_transform(table[1:])  # (N,d)->(N,pca)
        pca_table = np.vstack([np.zeros((1, args.pca_dim), dtype=np.float32), pca_emb.astype(np.float32)])
        out_pca = args.out_pca if args.out_pca else args.out.replace(".pkl", f"_pca{args.pca_dim}.pkl")
        pickle.dump(pca_table, open(out_pca, "wb"))
        print("Saved PCA table:", out_pca, pca_table.shape)

if __name__ == "__main__":
    main()
