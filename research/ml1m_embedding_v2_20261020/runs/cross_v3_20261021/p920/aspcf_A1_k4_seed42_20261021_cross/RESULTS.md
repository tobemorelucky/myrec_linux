# Crossed-asset ASPCF SS exploratory result

Variant A1; seed42, K4, H20, lr .001, batch1024, relation .01. Original V1 configuration except embedding/output/GPU.

Best/end epochs 76/96; from scratch; Dev chooses best, then one fixed-checkpoint Test.

|phase|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|
|---|---:|---:|---:|---:|---:|---:|
|Dev|0.374503311258|0.274247506426|0.484437086093|0.309868357592|0.591390728477|0.336957798729|
|Test|0.342052980132|0.249616597982|0.44917218543|0.284326164052|0.558278145695|0.311894642039|

Checkpoint SHA256: 65343d00f84bcfb4e8deb4fce47149f55da08cdfdd98c54ec77b0bcd1f26409b
Embedding SHA256: 03cf9649844da2c172487a0f552a49a4d331d8262135e3b70e48d6e772ad7905

Historical Test observations informed this research; this is exploratory, not an independent confirmatory test.