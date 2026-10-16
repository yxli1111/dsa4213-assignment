# DSA4213 Assignment 1 — Language Models, Embeddings, and Representations

This repository contains CPU-based experiments comparing next-token language models, input-embedding choices, and frozen language-model representations for downstream text classification.

## Project overview

- **Part I:** Interpolated add-\(k\) 4-gram, vanilla RNN, LSTM, and causal Transformer on WikiText-2 Raw.
- **Part II:** Transformer with (A) random trainable embeddings, (B) self-trained frozen GloVe, and (C) public frozen GloVe.
- **Part III:** Frozen RNN, LSTM, Transformer, self-GloVe Transformer, and public-GloVe Transformer representations for IMDb sentiment, AG News topic, and DBpedia-14 entity classification. A standardized logistic-regression classifier is trained on each representation.
- **Additional analyses:** matched text generation, error patterns, and unigram domain similarity.

All experiments use a training-derived 10,000-entry word vocabulary. The neural language models use 64-token independent windows, without cross-window recurrent state.

## Environment

The recorded environment was **Windows 10, Python 3.8.10, CPU-only PyTorch 2.0.1**, on a ThinkPad T14 with an AMD Ryzen 7 PRO 4750U and 16 GB RAM. Training scripts generally used eight CPU threads.

Install dependencies in a Python 3.8 virtual environment:

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`torch==2.0.1+cpu` requires the appropriate PyTorch CPU wheel source. If the ordinary installation cannot resolve it, install the CPU wheel using the official PyTorch wheel index, then install the remaining requirements. The supplied `requirements.txt` pins the main project dependencies, not every transitive package in the original environment.

Run all module commands **from the repository root**, using `.\.venv\Scripts\python.exe -m ...` on Windows. Notebook paths locate the project root by looking for `src/` and `results/`.

## Repository layout

```text
src/                 Data preparation, training, evaluation, generation and analysis
notebooks/           Exploratory notebooks and publication-ready figures/tables
data/part3/          Prepared downstream train/validation/test JSONL files
checkpoints/         Saved language models, n-gram counts and GloVe vectors
results/             Run-specific metrics, predictions, features and analyses
glove.6B/            Locally downloaded Stanford GloVe 6B vectors
prompts_part1_part2.json
EXPERIMENT_LOG.md    Experimental choices, checks and interpretations
```

`backups/` contains earlier training snapshots. Most `results/` and `checkpoints/` subfolders include multiple runs: use the specific run IDs below rather than assuming the newest directory is the selected model.

## Data access and preprocessing

**WikiText-2 Raw:** `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, using official train/validation/test splits, downloaded through Hugging Face `datasets` when the scripts run. The training split determines the word vocabulary and the numeric normalization rule. Preprocessing lowercases, splits on whitespace, normalizes WikiText markers (`@-@`, `@.@`, `@,@`), preserves article boundaries, inserts `<bos>`, `<eos>`, `<section>`, and `<para>`, and maps rare standalone ASCII integers to `<num>` (training frequency below 20). The seven reserved IDs are `<pad>` 0, `<unk>` 1, `<num>` 2, `<bos>` 3, `<eos>` 4, `<section>` 5, and `<para>` 6.

**Downstream data:** `imdb`, `ag_news`, and `fancyzhx/dbpedia_14` are downloaded through `datasets`. Each task uses a fixed stratified subset of 4,000 train, 1,000 validation, and 2,000 test examples, with seed 42. Validation examples are selected from the official training split; test examples come from the official test split. The downstream preparation scripts apply lowercasing, whitespace tokenization, WikiText marker conversion, and the WikiText training-derived vocabulary/number rule; IMDb HTML markup is cleaned. Documents are not truncated and are processed as independent 64-token chunks for frozen feature extraction.

DBpedia preparation excludes candidate overlaps identified by the title-based WikiText-2/DBpedia audit. This is a screening procedure, not a full text-level deduplication guarantee.

**Public embeddings:** Download Stanford **GloVe 6B, 100d** separately and place `glove.6B.100d.txt` at `glove.6B/glove.6B.100d.txt`. Public GloVe is an allowed external pretrained word embedding, not a pretrained language model. Alignment is restricted to the shared 10k vocabulary; unmatched ordinary tokens are randomly initialized, and the full embedding table remains frozen in Part II-C.

## Reproduction workflow

The commands below describe the source workflow. **Several evaluation and extraction scripts refer to the original selected run IDs**, so a clean retraining run with new timestamped IDs requires updating those references (or using the original trusted checkpoints). Data-preparation scripts write fixed paths under `data/part3/`; back up existing prepared data before rerunning them.

### Part I — Language models

Train the three neural models for the original first ten epochs:

```powershell
.\.venv\Scripts\python.exe -m src.run_experiments
```

The original experiment continued all three models for five additional epochs using `src.resume_experiments` and the saved epoch-10 checkpoints:

```powershell
.\.venv\Scripts\python.exe -m src.resume_experiments
```

For an individual fresh 15-epoch neural run, `src.train` exposes `--model`, `--epochs`, `--vocab-size`, `--context-length`, `--batch-size`, `--learning-rate`, `--threads`, `--seed`, and `--resume`. The shared settings were vocabulary 10,000, context 64, batch size 32, Adam learning rate 0.001, seed 42, and 15 epochs.

Train the count-based model and conduct validation-only smoothing searches:

```powershell
.\.venv\Scripts\python.exe -m src.train_ngram
.\.venv\Scripts\python.exe -m src.validate_ngram
.\.venv\Scripts\python.exe -m src.ngram_search_2
.\.venv\Scripts\python.exe -m src.ngram_search_3
```

The selected 4-gram uses add-\(k\) smoothing with `k=0.0001` and interpolation weights `[0.20, 0.45, 0.30, 0.05]` from unigram through 4-gram. Search scripts refer to the original n-gram run by default.

Verify validation and evaluate the original selected checkpoints on the held-out WikiText-2 test split:

```powershell
.\.venv\Scripts\python.exe -m src.check_part1_evaluation
.\.venv\Scripts\python.exe -m src.evaluate_part1
```

`evaluate_part1` refuses to overwrite existing final test outputs. Selected checkpoints:

| Model | Selected checkpoint |
|---|---|
| RNN | `checkpoints/rnn/20261014_002943_006765/best.pt` |
| LSTM | `checkpoints/lstm/20261014_013354_989299/best.pt` |
| Transformer | `checkpoints/transformer/20261014_023743_769122/best.pt` |
| 4-gram | `checkpoints/ngram/20261014_164502_309605/counts.pkl` |

Final results: `results/part1_final/test_results.csv`; model-size summary: `results/part1_model_size/20261015_180542_559429/model_size.csv`.

### Part II — Embedding variants

Train self GloVe on the WikiText-2 training split (the final embedding run used 20 epochs):

```powershell
.\.venv\Scripts\python.exe -m src.train_glove --epochs 20
```

Align public GloVe vectors to the fixed vocabulary:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_public_glove --glove-path "glove.6B/glove.6B.100d.txt"
```

Train the two frozen-embedding Transformer variants for 15 epochs:

```powershell
.\.venv\Scripts\python.exe -m src.run_part2 --epochs 15
```

Part II-A reuses the Part I Transformer. Part II-B and C train their remaining Transformer parameters from scratch while freezing the entire input embedding table. The training scripts use the selected embedding-export paths configured in source; if creating new embedding runs, check/update those paths first.

```powershell
.\.venv\Scripts\python.exe -m src.check_part2_evaluation
.\.venv\Scripts\python.exe -m src.evaluate_part2_test
```

Selected B and C checkpoints:

- `checkpoints/part2/self_glove/20261014_194751_117614/best.pt`
- `checkpoints/part2/public_glove/20261014_214138_156838/best.pt`

Final test results: `results/part2/test_evaluation/20261015_144104_950573/results.csv`.

### Text generation and analysis

The generation script accepts `--prompts`, `--decoding`, `--temperature`, `--top-k`, `--top-p`, `--new-tokens`, `--seed`, and `--ban-unk`. Two matched runs with banned generated `<unk>` were used for the final diversity/repetition analysis: temperature 0.8 sampling (top-k 40) and greedy decoding.

```powershell
.\.venv\Scripts\python.exe -m src.generate_comparisons --prompts prompts_part1_part2.json --decoding greedy --ban-unk
.\.venv\Scripts\python.exe -m src.analyze_generation_metrics --sample-csv "results/generation/20261015_142451_559597/generations.csv" --greedy-csv "results/generation/20261015_143114_234189/generations.csv"
```

The quantitative analysis is saved under `results/generation/quantitative_analysis/`.

An exploratory AI-assisted review of the 288 continuations (24 prompts × six models × two decoding settings) is archived in `results/ai_review/`. The full per-continuation record is `results/ai_review/generation_ai_review_completed_288.csv`; the same folder contains model/decoding and prompt-group summaries and the scoring methodology. These subjective scores supplement, rather than replace, the automatic diversity/repetition metrics and original generated text.

### Part III — Downstream representation classification

Prepare the IMDb and AG News fixed subsets, then audit DBpedia titles and prepare its fixed subset:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_part3_data
.\.venv\Scripts\python.exe -m src.audit_dbpedia_overlap_v2
.\.venv\Scripts\python.exe -m src.prepare_part3_dbpedia
```

The audit records candidate title matches with WikiText-2 training articles; DBpedia preparation excludes the candidate source indices.

Extract frozen token-mean hidden-state representations. The trusted-checkpoint flag is only appropriate for files created and trusted by the operator; PyTorch/pickle checkpoint deserialization can execute code.

```powershell
.\.venv\Scripts\python.exe -m src.extract_part3_representations --trust-checkpoints
.\.venv\Scripts\python.exe -m src.extract_part3_representations_with_self_glove --models self_glove --trust-checkpoints
```

The first command extracts RNN, LSTM, Transformer, and public GloVe; the second adds self GloVe. The original outputs are:

- Four-model features: `results/part3/representations/20261015_164420_564223/`
- Self-GloVe features: `results/part3/representations/20261015_165954_901056/`

Train/evaluate standardized logistic-regression classifiers with `C` selected from `{0.1, 1, 10}` on validation Macro-F1. The held-out test set is evaluated after model selection:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_part3_classification --features-dir "results/part3/representations/20261015_164420_564223"
.\.venv\Scripts\python.exe -m src.evaluate_part3_classification_with_self_glove --models self_glove --features-dir "results/part3/representations/20261015_165954_901056"
```

Original classification outputs:

- Four-model run: `results/part3/classification/20261015_165639_601149/`
- Self-GloVe run: `results/part3/classification/20261015_170432_954028/`

Post-hoc analysis of saved predictions and text distributions:

```powershell
.\.venv\Scripts\python.exe -m src.analyze_part3_errors
.\.venv\Scripts\python.exe -m src.analyze_domain_similarity
```

The error-analysis script combines the two original classification result directories. Domain similarity compares unigram distributions of WikiText-2 training text and the downstream **training subsets** using Jensen–Shannon divergence.

### Figures and tables

Run the notebooks from `notebooks/` with the project's Python kernel:

- `part1_learning_curves.ipynb`
- `part2_learning_curves.ipynb`
- `part3_results_table.ipynb`

Their saved outputs are under `results/part1_learning_curves/`, `results/part2_learning_curves/`, and `results/part3/report_tables/`.

## Recorded headline results

| Part I model | WikiText-2 test PPL |
|---|---:|
| 4-gram | 159.40 |
| RNN | 121.41 |
| LSTM | 117.55 |
| Transformer | **105.84** |

| Part II Transformer input embedding | WikiText-2 test PPL |
|---|---:|
| Random trainable | 105.84 |
| Self GloVe frozen | 119.86 |
| Public GloVe frozen | **105.27** |

| Frozen representation | IMDb test Macro-F1 | AG News | DBpedia |
|---|---:|---:|---:|
| RNN | 0.6844 | 0.6112 | 0.8035 |
| LSTM | 0.6865 | 0.6010 | 0.7983 |
| Transformer | 0.6730 | 0.6701 | 0.8505 |
| Self GloVe | 0.7050 | 0.6526 | 0.8342 |
| Public GloVe | **0.7199** | **0.8142** | **0.9053** |

The three downstream tasks have different label spaces and text characteristics; compare representation scores primarily **within** each task. These are results of one recorded experimental configuration, not a multi-seed estimate.

## Reproducibility details

- Training seed: **42**; downstream subset selection seed: **42**.
- WikiText vocabulary and numeric normalization are derived from training data only.
- Neural LM windows: context 64, stride 64; padding positions excluded from LM loss.
- Part III frozen feature extraction: all chunks retained; contextual token states averaged across the document; no cross-chunk hidden-state carryover.
- Classifier: `StandardScaler` fitted on downstream train only, followed by `LogisticRegression`; validation Macro-F1 selects regularization.
- Checkpoint paths and some evaluation scripts refer to original timestamped runs; inspect source/config files when reproducing from new runs.
- Training/evaluation cost depends on CPU, threading, and timing procedure. The saved WikiText-2 evaluation times are **teacher-forced evaluation**, not autoregressive generation speed.

## Data and tool attribution

- WikiText-2 Raw: Salesforce/Hugging Face dataset distribution, originally from the WikiText corpus.
- IMDb sentiment: Hugging Face `imdb` dataset.
- AG News: Hugging Face `ag_news` dataset.
- DBpedia-14: Hugging Face `fancyzhx/dbpedia_14`.
- Public word embeddings: Stanford GloVe 6B 100d.
- Core libraries: PyTorch, Hugging Face Datasets, NumPy, pandas, scikit-learn, SciPy, gensim, matplotlib.
- AI assistance: ChatGPT was used for code drafting/debugging, experiment workflow planning, and documentation support; the experimental choices, execution, validation, and interpretation should be reviewed and understood by the submitting author.

Dataset and embedding licenses/terms apply. Avoid publishing restricted dataset copies or secrets in a public repository.
