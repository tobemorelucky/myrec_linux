# ML-1M final CPU admission (2026-10-20)

This document supersedes the Base-SCFT draft in README_PRODUCTION.md. No GPU or production stage has run.

- Verified base: Meta Llama-2-7b-hf official public Git LFS pointer OIDs match locally hashed safetensors. The bin copy is tensor-identical; never delete without separate approval. Toys/ML/Beauty are separate LoRAs.
- Existing Rich overview/director coverage is zero. Existing enhanced prompts contain average ratings and are forbidden inputs. No bulk metadata was fetched.
- Preferred V2 is ONE Rich-SCFT, then SAME new LoRA encodes Base/Rich. It is BLOCKED until actual verified overview coverage >=90%, no included unresolved IDs, no cross-movie token collisions, augmented-view truncation <=5%. Thresholds are preregistered engineering gates, not performance tuning. No silent Base-SCFT fallback. Final Rich Prompt/config SHA CANNOT be frozen before data exists.
- Use official MovieLens links + same-release movies.csv + title/year/genre/external-ID checks, then TMDB overview/director/cast. Ambiguities stay Base. IMDb public TSVs have no plot; do not scrape IMDb plots. TMDB credentials are read ONLY from TMDB_READ_TOKEN; terms confirmation required. rich_metadata.py defaults to at most five explicitly approved sample calls; a separately approved catalog run additionally needs --approved-bulk and an exact item limit (<=3706), optional explicit --resume-existing. Full collection needs a separate scope decision; do not rent GPU while Rich is missing.
- Scripts and inputs remain in this existing independent directory. All pretrained 7B execution is rental-only; P920 allows CPU tests/tokenizer only.

## CPU metadata preparation (future commands, no collection executed)

```
python -B adapter/rich_metadata.py --movies /root/autodl-tmp/pomrec/data/ml-1m/item_str.jsonline --mapping /root/autodl-tmp/pomrec/data/ml-1m/item2id.json
# After official links.csv + matching movies.csv are present:
python -B adapter/rich_metadata.py --movies /root/autodl-tmp/pomrec/data/ml-1m/item_str.jsonline --mapping /root/autodl-tmp/pomrec/data/ml-1m/item2id.json --links <links.csv> --catalog-movies <movies.csv> --responses <approved_response_folder> --out <new_intake_folder> --execute
# For at most five official TMDB sample requests, add --fetch-samples 3 --confirm-tmdb-noncommercial-terms, and set the credential privately in the environment.
python -B adapter/prepare_prompts.py --movies /root/autodl-tmp/pomrec/data/ml-1m/item_str.jsonline --mapping /root/autodl-tmp/pomrec/data/ml-1m/item2id.json --rich <new_intake_folder>/rich_verified.json --out inputs/rich_prompts.jsonl --execute
python -B adapter/freeze_rich.py --prompt inputs/rich_prompts.jsonl --coverage <new_intake_folder>/coverage.json --execute
```

## Separately approved GPU stages

Rental working directory: /root/autodl-tmp/pomrec/research/ml1m_embedding_v2_20261020

```
GPU_INDEX=0 bash scripts/gpu_stages.sh A
# Stop and review A. Then individually authorized:
GPU_INDEX=0 bash scripts/gpu_stages.sh B
GPU_INDEX=0 bash scripts/gpu_stages.sh C
# C uses 3 optimizer steps, full batch16/accum8, real registered Rich input, new temporary LoRA.
# D requires user confirmation after C; starts from the same BASE from scratch, NOT C checkpoint:
V2_SCFT_APPROVED=YES V2_WALL_SECONDS_APPROVED=14400 GPU_INDEX=0 bash scripts/gpu_stages.sh D
GPU_INDEX=0 bash scripts/gpu_stages.sh E
bash scripts/gpu_stages.sh F
```

A: fixed eight canonical IDs; SAMPLE_COMPLETE cannot be used as full PCA input. B: full old-LoRA V1-Rebuilt raw, Original unchanged. C: same official Trainer/model/soft-target loss, finite gradients + adapter/tau round trip; no second full model. D: 1000-step only-once Rich SCFT. E: Base/Rich raw + common Train-catalog PCA; independent V1-Rebuilt PCA. F: hashes only; transfer/backup must be separately scoped and verified. No asset replacement or deletion.

SCFT fixed r8/alpha32/dropout.1, microbatch16 items (32 views), accumulation8, lr2e-4, 1000 optimizer steps, seed42, original soft-target loss, FP16 AMP + FP32 similarity/pooling/tau. Normal negatives15, tail9; accumulation is NOT 127 contrastive negatives. Title-preserving real-attribute dropout .4/shuffle, keep >=2 attributes; cap overview/plot160, director24, cast24 before BOTH training augmentation and encoding; total256 and EOS. These caps await validation on real Rich. Original V1 max64; original LoRA is never overwritten.

Training saves adapter+tau+tokenizer+optimizer/scheduler/RNG/Trainer state, never full base. Local safetensors-only/offline loading. Errors stop; no automatic batch/precision changes. All runs use new paths, no auto resume/retries. Stage D requires SAME config as successful C.

GPU planning only: prefer48GB; 32GB is possible subject to real C peak, no guarantee. Encoding needs >=14GiB free, SCFT precheck >=24GiB. Keep >=8GiB disk reserve and budget <=4GiB new artifacts before backup; current free about13.6GiB. Actual GPU timings unknown. No GPU allocated at CPU review.

Final regression also protects the actual MovieID3845 title HTML entity semicolon. Original text remains unchanged; only attribute delimiter parsing is corrected. CPU suite now contains32 tests. Release03 is superseded by04.

Original Base JSONL physical row order is not canonical item-ID order. It is valid: identity is verified via item/item_id mapping, not physical line number. Encodes emit canonical order; SCFT and its augmentation audit use the actual immutable Prompt-file row order. Final suite contains34 tests; releases03/04 are superseded by05.

Stage D also requires the separately approved V2_WALL_SECONDS_APPROVED; timeout sends INT then KILL after60s, with no automatic retry. Rich length may multiply old Base training cost: estimate 1000 * measured C seconds/update, not the old44.5-minute number. Do not start paid D until time/memory margin is reviewed.
