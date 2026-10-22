# P920 crossed-assets entry points
Original scaled/generic launchers and all public models/Reader/Runner remain unchanged.
Reuse base.build/bind/smoke/dev/supervise from the original V2 launcher loaded via the original scaled launcher. Exact one-line in-memory adaptation permits the same GPU as the V1 reference while requiring a distinct embedding. No global rescaling; no activation probes; original V1 config copied completely except embedding/output/GPU.
Every matrix is independently verified SHA/shape [3707,1536] float32/padding/finite/nonzero/no duplicate rows. Producer identity and canonical order are manifest attestations, retained verbatim.

Paths relative to research/ml1m_embedding_v2_20261020:
- A1: runs/cross_v3_20261021/p920/incoming/A1_oldlora_rich512/rich_pca.pkl
- B0: runs/cross_v3_20261021/p920/incoming/B0_v2lora_base/base_pca.pkl
- V3: runs/cross_v3_20261021/p920/incoming/V3/rich_pca.pkl

Write READY.json last, after the matrix and full producer manifest are safely transferred and SHA confirmed. READY required keys:
{"variant":"A1","embedding_sha256":"ACTUAL_64_LOWERCASE_HEX","canonical_item_ids":"1..3706","lora_family":"old","text_recipe":"rich512","run_name":"aspcf_A1_seed42_UNIQUE_TIMESTAMP","producer_manifest":{...}}
B0 identity: lora_family=v2, text_recipe=base_original. V3 identity: lora_family=v3, text_recipe=the_final_selected_recipe.
Additional complete producer SHA/config fields can be embedded; they are preserved verbatim.

Explicit prepare (no training):
/home/yyx/anaconda3/envs/hzg_py10/bin/python -B research/ml1m_embedding_v2_20261020/ops/cross_v3_20261021/p920/run_cross_aspcf.py --mode prepare --variant A1 --gpu 0 --run-name UNIQUE_RUN_NAME --embedding-sha256 ACTUAL_SHA --delivery-manifest ABSOLUTE_READY_PATH
Then validate or supervise:
/home/yyx/anaconda3/envs/hzg_py10/bin/python -B research/ml1m_embedding_v2_20261020/ops/cross_v3_20261021/p920/run_cross_aspcf.py --mode validate --run-dir ABSOLUTE_RUN_DIR
/home/yyx/anaconda3/envs/hzg_py10/bin/python -u -B research/ml1m_embedding_v2_20261020/ops/cross_v3_20261021/p920/run_cross_aspcf.py --mode supervise --run-dir ABSOLUTE_RUN_DIR

Optional queue is not started during preparation. Once explicitly launched it waits for each READY.json, validates identity/SHA/config, and waits for two idle checks. A1 GPU0; B0 GPU1; V3 any idle GPU. No preemption, no chat notifications, no duplicate/existing-run replacement. Queue wait expires after 24h by default; launched training retains its original 4h wall limit.
/home/yyx/anaconda3/envs/hzg_py10/bin/python -u -B research/ml1m_embedding_v2_20261020/ops/cross_v3_20261021/p920/asset_queue.py --queue-name UNIQUE_TIMESTAMP --wait-hours 24

Successful complete training -> Dev selection -> one fixed-best independent Test. Original 1001 slots, duplicate preservation, original Runner six metrics @5/10/20. Nonfinite losses/checkpoint, source/data/config changes, failed training or smoke prevent Test. TEST_STARTED.json is exclusive and blocks automatic Test retry. Previously observed Test makes all downstream research exploratory.
No startup currently occurs just from creating receiving directories or this documentation.
