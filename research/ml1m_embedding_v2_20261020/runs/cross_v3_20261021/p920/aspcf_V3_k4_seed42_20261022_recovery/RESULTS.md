# Crossed-asset ASPCF SS exploratory result

Variant V3; seed42, K4, H20, lr .001, batch1024, relation .01. Original V1 configuration except embedding/output/GPU.

Best/end epochs 55/75; from scratch; Dev chooses best, then one fixed-checkpoint Test.

|phase|HR@5|NDCG@5|HR@10|NDCG@10|HR@20|NDCG@20|
|---|---:|---:|---:|---:|---:|---:|
|Dev|0.366887417219|0.267616294402|0.472847682119|0.301911174573|0.576986754967|0.328303620746|
|Test|0.330298013245|0.241528064135|0.433278145695|0.274868204987|0.537582781457|0.301247469551|

Checkpoint SHA256: e66cf0e0ac0b1fd3af8d9b48055b23064b9f511e735e273ff2020cb3f7417ee8
Embedding SHA256: b393145c5520c7018c751938cec01a834ccdd8fe3df1626673ea62a1c8179700

Historical Test observations informed this research; this is exploratory, not an independent confirmatory test.