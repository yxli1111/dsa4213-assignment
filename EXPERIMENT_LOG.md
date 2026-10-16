# DSA4213 Assignment 1 — Final Experiment Decision Log

## Data selection and evaluation protocol

Training-derived vocabulary and numeric rules were fixed before validation-based model selection, and official test splits were reserved for final evaluation.

## Detailed experimental decisions

### Decision 01 — Vocabulary and tokenization

The 10,000-token vocabulary was selected over an 8,000-token alternative after CPU benchmarking and training-corpus coverage checks.

Ordinary training-token OOV fell from 9.56% with 8,000 entries to 7.86% with 10,000 entries.

The vocabulary was constructed from the training split only, with frequency ordering and lexicographic tie-breaking.

Subword tokenization was not selected because the word-level pipeline aligned directly with GloVe and kept the model comparison consistent.

The numeric frequency threshold of 20 retained frequently observed standalone integers while mapping rarer integers to <num>.

WikiText punctuation markers were normalized as standalone tokens; otherwise, ordinary punctuation remained subject to whitespace tokenization.

### Decision 02 — Article structure and training windows

Level-one headings started new articles, later headings introduced <section>, and nonempty body records were preceded by <para>.

We compared 32/32, 32/16, and 64/64 context-length/stride configurations; all covered 2,046,135 distinct training targets.

The 64/64 configuration was selected to provide longer local context without overlapping target predictions and with manageable CPU cost.

Independent windows reset RNN/LSTM hidden states, while the Transformer used a causal attention mask; no model received information from future targets.

### Decision 03 — Training budget and checkpoint selection

An initial 10-epoch neural training budget was extended equally to 15 epochs because validation losses were still decreasing.

RNN, LSTM, and Transformer initial 10-epoch wall times were 64.20, 63.81, and 73.20 minutes; the additional five epochs took 31.58, 31.84, and 34.32 minutes.

The best checkpoint was selected using validation loss, rather than choosing the last epoch automatically.

The causal-mask future-token perturbation test reported zero change in earlier-position logits at displayed precision.

### Decision 04 — n-gram smoothing search

The n-gram baseline used interpolated Add-k probabilities up to order four, with no neural parameters.

Three validation-only search rounds tested 27, 24, and 72 configurations, reducing the best validation perplexity from 185.55 to 172.47 and then 170.3159.

The final smoothing value was k=0.0001, with unigram-to-four-gram interpolation weights [0.20, 0.45, 0.30, 0.05].

### Decision 05 — Embedding variants

Part II reused the Part I Transformer for the random-trainable condition, keeping the other architecture and training settings aligned for the two frozen-embedding runs.

Self-trained GloVe used a context radius of five, inverse-distance weighting, x_max=100, alpha=0.75, AdaGrad learning rate 0.05, and 20 epochs.

The final 20-epoch self-GloVe run restarted from random initialization rather than continuing the earlier exploratory 10-epoch run.

The public GloVe 6B 100-dimensional vectors were aligned to the fixed WikiText-derived vocabulary, without enlarging the prediction vocabulary.

The 86 unmatched ordinary vocabulary entries were initialized with seeded Normal(0, 0.1) vectors and frozen together with all matched entries.

### Decision 06 — Generation and decoding

The prompt set crossed short and long contexts, wiki-like and non-wiki-like language, and simple and moderately complex vocabulary.

Four exploratory decoding runs were recorded: unrestricted temperature-1.0 sampling, temperature-0.8 top-k-40 sampling, the same sampling with generated <unk> banned, and greedy decoding with generated <unk> banned.

The final quantitative comparison used the last two conditions, each with 24 matched prompts, six models, and a 50-token continuation limit.

### Decision 07 — Downstream task selection

IMDb sentiment, AG News topic classification, and DBpedia-14 ontology classification were retained to compare representations across different text sources and label structures.

Length exploration showed IMDb reviews were substantially longer than AG News articles; full texts were therefore processed as independent 64-token chunks without truncation.

Each downstream task used class-stratified subsets of 4,000 training, 1,000 validation, and 2,000 test samples.

A two-stage DBpedia title-overlap audit was followed by exclusion of 181 training and 29 test candidates before sampling.

The downstream vocabulary and numeric normalization remained fixed from Part I; no downstream dataset was used to rebuild the vocabulary.

### Decision 08 — Frozen representations and classification

All five language models were frozen, and document features were computed by averaging valid token-level hidden states over all chunks.

A StandardScaler plus Logistic Regression probe was trained for each representation and task; C was selected from {0.1, 1, 10} on validation macro-F1.

The three official test subsets were evaluated after hyperparameter selection, and error cases were joined by source index for paired comparisons.

### Decision 09 — Domain similarity follow-up

To examine whether downstream text resembled WikiText-2, unigram Jensen–Shannon divergence was computed using WikiText-2 training text and each downstream training subset.

DBpedia had the smallest measured divergence (0.32011 bits), followed by IMDb (0.33411) and AG News (0.33703).

## Part I — Corpus and preprocessing

The shared language-modeling corpus was Salesforce WikiText-2 Raw (wikitext-2-raw-v1), using the official train, validation, and test splits.

The training split was parsed into 629 article-bounded sequences with 2,046,764 processed token IDs and 2,046,135 next-token targets per epoch.

Text was lowercased and whitespace-tokenized; standalone WikiText markers @-@, @.@, and @,@ were mapped to -, ., and , respectively.

Article beginnings and endings used <bos> and <eos>; section headings and body records used <section> and <para>.

ASCII integer tokens occurring fewer than 20 times in the training corpus were mapped to <num>, and the training-derived rule was reused on later splits.

The final word-level vocabulary contained 10,000 entries including seven reserved tokens, and ordinary-token training OOV was approximately 7.86%.

Neural training used article-bounded windows of 64 input tokens with stride 64, resetting recurrent states between windows and masking padding and duplicated targets from loss.

This window protocol limits cross-window context and is not a stateful or full-document recurrent training setup.

## Part I — Architectures, selection, and prediction quality

The compared models were an interpolated Add-k 4-gram, a one-layer vanilla RNN, a one-layer LSTM, and a two-layer causal Transformer.

All three neural models used randomly initialized trainable 100-dimensional token embeddings and the same 10,000-token prediction vocabulary.

RNN and LSTM hidden sizes were 128; the Transformer used four attention heads, a 256-dimensional feed-forward layer, and learned positional embeddings.

Neural training used Adam with learning rate 0.001, gradient clipping at 1.0, batch size 32, eight CPU threads, and seed 42.

Each neural model completed 15 epochs, 30,692,025 cumulative training targets, and 15,135 optimizer steps.

Best checkpoints were selected by validation loss: RNN at epoch 12, LSTM at epoch 15, and Transformer at epoch 15.

The 4-gram used k = 0.0001 and interpolation weights [0.20, 0.45, 0.30, 0.05] for orders one through four, selected through validation-only searches.

| Model | Best validation PPL | Test cross-entropy | Test PPL | Selected epoch |
|---|---:|---:|---:|---:|
| 4-gram | 170.3159 | 5.071390 | 159.3957 | N/A |
| RNN | 132.1279 | 4.799211 | 121.4146 | 12 |
| LSTM | 127.6028 | 4.766825 | 117.5455 | 15 |
| Transformer | 114.4093 | 4.661910 | **105.8381** | 15 |

All four models were evaluated on the same 240,628 held-out next-token targets, but the n-gram used up to three article-continuous preceding tokens whereas neural models reset at 64-token boundaries.

Transformer achieved the lowest held-out perplexity under this protocol, without establishing a universal ranking of model families.

## Part I — Cost, model size, and learning behavior

| Model | Parameters | Weight size (MiB) | Checkpoint size (MiB) | Test evaluation (s) |
|---|---:|---:|---:|---:|
| RNN | 2,319,440 | 8.8480 | 26.7579 | 20.08 |
| LSTM | 2,407,760 | 9.1849 | 27.7691 | 20.47 |
| Transformer | 2,207,512 | 8.4210 | 25.4987 | 19.55 |
| 4-gram | N/A | N/A | 53.7235 | 2.15 |

The 4-gram checkpoint stored 3,365,795 distinct n-gram entries; its serialized size is not equivalent to neural weight memory or live RAM use.

RNN, LSTM, and Transformer two-session training wall times were approximately 95.79, 95.65, and 107.52 minutes respectively, including preparation, validation, and checkpoint overhead.

The n-gram count-building stage took 12.49 seconds, excluding preparation, saving, and validation search, so this is not an end-to-end training-time comparison.

Evaluation times are teacher-forced test workload timings rather than autoregressive generation latency; the measured workload and hardware must be stated in the report.

The recorded CPU environment was a ThinkPad T14 with AMD Ryzen 7 PRO 4750U and 16 GB RAM, using Windows and PyTorch CPU execution.

RNN validation loss reached its minimum at epoch 12 and then rose slightly, while LSTM and Transformer validation losses continued decreasing through epoch 15.

The last recorded train–validation gaps were 0.4552 for RNN, 0.3190 for LSTM, and 0.4071 for Transformer; these gaps alone do not quantify generalization quality.

## Part II — Controlled embedding variants

Part II reused the Part I Transformer as setting A and trained two additional Transformers from scratch above fixed input embeddings.

Setting A used randomly initialized trainable embeddings, setting B used self-trained frozen GloVe, and setting C used public frozen GloVe 6B 100d.

All variants shared the 10,000-entry vocabulary, 100-dimensional embeddings, Transformer architecture, training data, 64-token window protocol, and 15-epoch budget.

The self-trained GloVe used training-only co-occurrences with radius 5, 100 dimensions, AdaGrad learning rate 0.05, and 20 epochs.

Its reconstruction loss fell from 0.058323 at epoch 1 to 0.009289 at epoch 20, with approximately 4.9 minutes of optimizer training across the 20 epochs, excluding preprocessing and co-occurrence construction.

Public GloVe matched 9,907 of 9,993 ordinary vocabulary types (99.14% type coverage), leaving 86 unmatched ordinary tokens.

Unmatched public vectors were initialized with seeded Normal(0, 0.1), the padding vector was zeroed, and the entire embedding table remained frozen.

| Variant | Best val loss | Val PPL | Test loss | Test PPL | Last train loss |
|---|---:|---:|---:|---:|---:|
| A: Random trainable | 4.739782 | 114.4093 | 4.661910 | 105.8381 | 4.332731 |
| B: Self GloVe frozen | 4.864071 | 129.5506 | 4.786347 | 119.8628 | 4.623550 |
| C: Public GloVe frozen | **4.723505** | **112.5621** | **4.656501** | **105.2671** | 4.514417 |

All three validation curves were still improving at epoch 15, and the final-five-epoch validation-loss standard deviations were 0.01265, 0.00755, and 0.00876 for A, B, and C.

These standard deviations reflect within-run trends and fluctuations, not stability across independent seeds.

Public GloVe had a small held-out perplexity advantage over A, while B performed worse; external pretraining data and embedding trainability prevent a single-factor causal attribution.

### Decision 10 — Exploratory AI-assisted generation review

All 288 final continuations were assigned exploratory scores for repetition, local fluency, topic consistency, and internal coherence (0–5 per dimension, equally weighted). The review used a single AI evaluator, and most continuations were judged from abbreviated opening excerpts. The resulting scores are descriptive and are not a validated model ranking. The full records and summary tables are stored in `results/ai_review/`, including `generation_ai_review_completed_288.csv` and `generation_ai_review_methodology.md`.

## Parts I–II — Matched text generation

Six models were evaluated on 24 matched prompts, including short versus long contexts, wiki-like versus non-wiki-like styles, and Singapore haze-related hypothetical prompts.

The final compared decoding conditions were temperature 0.8 with top-k 40 and greedy decoding, both banning newly generated <unk> tokens and limiting continuation length to 50 tokens.

| Model | Greedy Distinct-2 | Greedy repeated-3 | Temp 0.8 Distinct-2 | Temp 0.8 repeated-3 |
|---|---:|---:|---:|---:|
| 4-gram | 0.374 | 0.615 | 0.951 | 0.006 |
| RNN | 0.580 | 0.346 | 0.956 | 0.003 |
| LSTM | 0.462 | 0.483 | 0.948 | 0.006 |
| Transformer | 0.531 | 0.350 | 0.940 | 0.010 |
| Self GloVe | 0.461 | 0.408 | 0.915 | 0.012 |
| Public GloVe | 0.495 | 0.384 | 0.935 | 0.008 |

Greedy decoding showed much more repeated-trigram behavior than temperature-0.8 sampling for all six models.

The generation metrics were computed from displayed continuation text and are approximate text-token statistics rather than exact original-token-ID metrics.

Distinct-n and repeated-n measure surface diversity and repetition, not factual correctness, fluency, or semantic coherence.

Most continuations reached the 50-token cap without the article-level <eos> marker; this does not imply that models cannot end sentences.

## Part III — Datasets, leakage screening, and representations

The three downstream tasks were IMDb binary sentiment classification, AG News four-class topic classification, and DBpedia-14 entity-type classification.

Each task used 4,000 training, 1,000 validation, and 2,000 test examples from the official source splits, with approximately balanced class quotas and seed 42.

Downstream texts were lowercased, whitespace-tokenized, and mapped to the fixed Part I vocabulary; unseen standalone ASCII integers were mapped to <num>.

The final training-set OOV rates were 22.67% for IMDb, 25.85% for AG News, and 25.18% for DBpedia.

Full downstream texts were retained without truncation and split into independent 64-token chunks with one <bos> token per document.

DBpedia input concatenated title and content, which may provide explicit clues about entity classes.

A WikiText-2 train versus DBpedia title audit flagged 181 official training rows and 29 official test rows, all of which were excluded before downstream sampling.

Selected DBpedia splits were also checked for exact duplicates of normalized title-plus-content text; title screening does not prove absence of all text overlap.

Frozen hidden states were extracted from RNN, LSTM, and three Transformer variants, then pooled as a token-weighted mean over all non-padding token states across chunks.

The resulting document vectors were 128-dimensional for RNN/LSTM and 100-dimensional for Transformer variants.

A StandardScaler and Logistic Regression classifier were trained on each frozen representation; validation macro-F1 selected C from {0.1, 1, 10} before one held-out test evaluation.

## Part III — Final held-out classification results

| Representation | IMDb Macro-F1 | AG News Macro-F1 | DBpedia Macro-F1 |
|---|---:|---:|---:|
| RNN | 0.6844 | 0.6112 | 0.8035 |
| LSTM | 0.6865 | 0.6010 | 0.7983 |
| Random Transformer | 0.6730 | 0.6701 | 0.8505 |
| Self GloVe Transformer | 0.7050 | 0.6526 | 0.8342 |
| Public GloVe Transformer | **0.7199** | **0.8142** | **0.9053** |

Public GloVe achieved the highest macro-F1 on all three downstream tasks under the fixed-feature linear-probe protocol.

Self GloVe outperformed the random Transformer on IMDb despite having worse language-model perplexity, showing that language-model and downstream rankings need not coincide.

Within-task comparisons are directly interpretable; cross-task macro-F1 differences cannot isolate domain similarity because label spaces and task structures differ.

## Part III — Error and domain-similarity analyses

| Task | Public fixes vs random Transformer | Public regressions | Net fewer errors |
|---|---:|---:|---:|
| IMDb | 297 | 203 | 94 |
| AG News | 375 | 89 | 286 |
| DBpedia | 184 | 75 | 109 |

AG News showed notable Business versus Sci/Tech confusion, while DBpedia included Animal versus Plant confusion; specific causal explanations require inspection of source texts.

IMDb examples included positive reviews containing negative descriptions or frustration about distribution, illustrating that sentiment targets can be ambiguous to linear probes.

Representative candidate IDs retained for qualitative verification include IMDb 278 and 20958, AG News 7158 and 4402, and DBpedia 13906 and 26541.

The error-case selection is illustrative rather than random, so examples do not estimate population-level error-type frequencies.

| Downstream training subset | JSD vs WikiText-2 train (bits) | WikiText-seen token mass |
|---|---:|---:|
| IMDb | 0.33411 | 83.44% |
| AG News | 0.33703 | 83.94% |
| DBpedia | **0.32011** | 81.45% |

DBpedia had the lowest unigram Jensen–Shannon divergence from WikiText-2 train, consistent with—but not proving—a possible relationship between lexical-domain proximity and transfer performance.

JSD was computed on downstream training subsets only and measures unigram frequencies rather than semantics, syntax, or causal transfer effects.

## Reproducibility paths and final reporting checklist

Core Part I sources include src/preprocessing.py, src/dataset.py, src/models.py, src/train.py, src/ngram.py, src/train_ngram.py, and src/evaluate_part1.py.

Part II sources include src/train_glove.py, src/prepare_public_glove.py, src/train_embedding_variant.py, and src/evaluate_part2_test.py.

Part III sources include the preparation, representation extraction, classification evaluation, error analysis, and domain-similarity scripts used in the recorded runs.

Recorded final outputs include results/part1_final, results/part2/test_evaluation, results/part3/classification, and results/part3/domain_similarity.

Generation comparison files are stored under results/generation, and final model-size measurements under results/part1_model_size.

Software versions, final script filenames, and reproduction commands are maintained in the project README.

The final report brings together learning curves, computational costs, generated-text examples, held-out results, and limitations.

The README documents data access, preprocessing, dependencies, execution commands, configurations, and random seeds.

Dataset and tool acknowledgments cover WikiText-2, IMDb, AG News, DBpedia-14, public GloVe, adapted external code, and substantial AI assistance.

The final documentation includes dataset and embedding citations and traceable sources for qualitative examples.
