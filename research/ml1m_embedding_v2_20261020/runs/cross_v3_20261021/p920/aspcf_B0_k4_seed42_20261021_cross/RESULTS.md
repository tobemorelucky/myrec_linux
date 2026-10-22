# Crossed-asset ASPCF SS exploratory result

Variant B0; seed42, K4, H20, lr .001, batch1024, relation .01. Original V1 configuration except embedding/output/GPU.

Best/end epochs 35/55; from scratch; Dev chooses best, then one fixed-checkpoint Test.

|phase|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|
|---|---:|---:|---:|---:|---:|---:|
|Dev|0.362417218543|0.261577000166|0.470529801325|0.29660883685|0.576986754967|0.323563650507|
|Test|0.326821192053|0.238611890734|0.435430463576|0.273612981953|0.545529801325|0.301360249881|

Checkpoint SHA256: 9b91b25d6ee7aa5d63424b9de4bea6b7b6aa5fa746b40583c7c12c265a3c76fa
Embedding SHA256: 70630546070652cd926fa13e9eff89de7e052d34e9ad0cfbc72b03376df702af

Historical Test observations informed this research; this is exploratory, not an independent confirmatory test.