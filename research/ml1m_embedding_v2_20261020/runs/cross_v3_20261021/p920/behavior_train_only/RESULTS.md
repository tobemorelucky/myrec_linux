# Train-only semantic-neighbor / behavior association

CPU elapsed 3.996s; Train rows 988129, users 6040, catalog 3704; fixed query sample 512; top10. No Dev/Test labels read.

|Embedding/control|Adjacent transition present|Transition degree cosine|Transition lift|Co-user cosine|Co-user lift|Target frequency|
|---|---:|---:|---:|---:|---:|---:|
|V1/semantic|0.254687|0.00360676|39.1931|0.203008|15.533|302.251|
|V1/uniform|0.0433594|0.000195563|0.970606|0.0896401|5.47236|271.985|
|V1/popularity_matched|0.0574609|0.000280416|2.28311|0.107402|7.27725|304.183|
|V1/semantic_minus_popularity_matched|0.197227|0.00332634|36.91|0.0956061|8.25579|-1.93195|
|V2/semantic|0.127344|0.00194817|19.4152|0.146425|9.73011|281.178|
|V2/uniform|0.0433594|0.000195563|0.970606|0.0896401|5.47236|271.985|
|V2/popularity_matched|0.04875|0.000231505|1.34631|0.0986652|6.42014|280.866|
|V2/semantic_minus_popularity_matched|0.0785937|0.00171667|18.0689|0.0477603|3.30998|0.31207|

Pair CSV contains every sampled semantic edge. JSON gives exact formulas, input SHA, fixed seeds, and paired query-bootstrap differences.

- Only Train structure association, not independent predictive evidence; no Dev/Test labels read.
- 512 fixed uniformly sampled Train-catalog queries; 10 semantic cosine neighbors, not an all-item benchmark.
- Adjacency preserves timestamp ties in CSV order, and is not proof of causal or next-item generalization.
- Degree/popularity adjustment reduces but cannot remove confounding; popularity-decile matching remains approximate.
- Query bootstrap intervals are descriptive; shared users/items and repeated-neighbor dependencies are not independent.
- No retraining, GPU, cache, corpus, checkpoint, or original-asset writes.