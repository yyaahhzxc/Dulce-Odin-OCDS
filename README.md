# Dulce-Odin-OCDS: An Ex-Ante Semantic Conflict Detection System for Davao City Ordinances

**Institution:** Ateneo de Davao University, Department of Computer Science  
**Authors:** Ralph Paolo Dulce & Yahyah Odin  
**Adviser:** Mr. Adrian "Ogs" Ablazo | **Professor:** Ma'am Grace Tacadao  
**Project:** *"A Coarse-to-Fine Semantic Conflict Detection System for Ex-Ante Davao City Ordinances Using Information Retrieval and Natural Language Inference"*

---

## Overview

**Dulce-Odin-OCDS** is an open-source, ex-ante semantic conflict detection system designed to audit draft municipal ordinances for legal inconsistencies prior to legislative enactment. Operating within the Philippine legal framework under the **Magtajas v. Pryce Properties** doctrine and Republic Act No. 7160 (Local Government Code of 1991), the system systematically flags both:
1. **Vertical Conflicts (Statutory Preemption):** Inconsistencies where a draft local ordinance permits what a superior national statute forbids, or forbids what a national statute permits.
2. **Horizontal Conflicts (Intra-Jurisdictional Coherence):** Inconsistencies where a draft ordinance contradicts, duplicates, or inadvertently causes an implied repeal of existing Davao City ordinances.

---

## System Architecture

The system implements a coarse-to-fine two-stage neural pipeline:

```
[Draft Ordinance Text or Scanned PDF]
                 │
                 ▼
[Document Parsing & Sectional Decomposition]
                 │
                 ▼
[Stage 1: Dual-Stream Hybrid Candidate Retrieval]
  ├── Sparse Lexical Stream: BM25 (Rank-BM25)
  └── Dense Semantic Stream: all-MiniLM-L6-v2 (Embedding Bi-Encoder)
  └── Convex Combination: Score = 0.40 * BM25_norm + 0.60 * Dense_norm
  └── Top-k Shortlist (k = 20) over 176,421 searchable dual-statutory provisions
                 │
                 ▼
[Stage 2: Cross-Encoder Natural Language Inference (NLI)]
  ├── Architecture: DeBERTa-v3-base Cross-Encoder (Full bidirectional cross-attention)
  ├── Input Formulation: [CLS] Statutory Provision [SEP] Draft Ordinance Section [SEP]
  └── Calibrated Decision Threshold: tau* = 0.45
                 │
                 ▼
[Explainable AI (XAI) Output & Audit Trail]
  ├── Verdict: Conflict / Contradiction (Preemption Alarm) vs. Neutral / Entailment
  ├── Softmax Calibrated Contradiction Probabilities & Margin Metrics
  ├── Attention Saliency Overlays & Verbatim Statutory Grounds
  └── Top-50 Candidate Audit Trail with Direct Lawphil URLs
```

### Key Architectural Tenets
- **Pure Neural Cross-Attention (Option 2):** All preemption and contradiction verdicts stem strictly from genuine PyTorch neural tensors. No regex heuristics, case-sniffing, hardcoded score dictionaries, or candidate injection backdoors are employed.
- **Zero Editorial Pollution:** All statutory texts, benchmark premises, and candidate provisions are 100% verbatim authentic text directly extracted from Lawphil and the Davao City Sangguniang Panlungsod archives.
- **Hierarchical Statutory Prepending:** Provisions are contextualized with their canonical enactment titles and section headers to maintain legal grounding during retrieval and inference.

---

## Dual Statutory Corpus

The underlying knowledge base compiles **27,096 legislative enactments** yielding **176,421 searchable provision-level retrieval units**:

| Jurisdiction | Legislative Scope | Enactments | Searchable Provisions | Operative Provisions |
| :--- | :--- | :--- | :--- | :--- |
| **National Statutes** | Acts, Commonwealth Acts, Republic Acts, Batas Pambansa, Presidential Decrees, Executive Orders | 25,432 | 164,620 | 164,620 (100.0%) |
| **Davao City Ordinances** | Sangguniang Panlungsod Local Ordinances (1945–2026) | 1,664 | 11,801 | 8,160 (69.1%) |
| **Total Dual Corpus** | **Combined Legislative Knowledge Base** | **27,096** | **176,421** | **172,780 (97.9%)** |

*Note: 3,641 local provisions represent legislative boilerplate clauses (Title, Scope, Separability, Repealing, and Effectivity clauses) isolated during decomposition.*

---

## Repository Structure

```
Dulce-Odin-OCDS/
├── corpus/
│   ├── national_laws/               # National statutory corpus JSONL files
│   │   ├── acts.jsonl
│   │   ├── batas_pambansa.jsonl
│   │   ├── commonwealth_acts.jsonl
│   │   ├── executive_orders.jsonl
│   │   ├── presidential_decrees.jsonl
│   │   └── republic_acts.jsonl
│   └── city_ordinances/             # Davao City municipal ordinance corpus
│       └── categorized_davao_ordinances.jsonl
├── data/
│   ├── blocks/                      # 5-fold cross-validation multi-panel blocks
│   ├── ocr_benchmark_samples/       # Archival OCR vs. VLM evaluation pairs
│   └── review_queue/                # Quality assurance and audit review transcripts
├── ground_truth/
│   ├── 350_pairs/                   # Multi-panel legal researcher benchmark (N = 350)
│   ├── tier2_synthetic_drafts/      # Full-document open-corpus triage drafts (6 synthetic ordinances)
│   └── tier3_sc_cases/              # 11 landmark Supreme Court preemption precedents
├── output/                          # Lightweight demo indices, registries & benchmark summaries
├── scripts/
│   ├── benchmarks/                  # Evaluation harnesses (retrieval, NLI screening, latency)
│   ├── corpus_build/                # Master dual-corpus compilation and normalization tools
│   ├── figures/                     # 300-DPI publication figure generation scripts
│   └── ground_truth_tools/          # Evaluation metrics, McNemar tests & Wilson CIs
├── src/
│   ├── demo/                        # Offline interactive web prototype (Flask + SSE)
│   │   ├── app.py                   # Server engine and REST/SSE endpoints
│   │   ├── canonical_hierarchies.py # Hierarchical prepending metadata
│   │   ├── sample_cases/            # 11 benchmark sample draft ordinances
│   │   └── static/                  # Responsive CSS, JavaScript UI and assets
│   ├── config.py                    # Global hyperparameters and path registry
│   ├── cross_encoder_nli.py         # Stage 2 DeBERTa-v3 inference engine
│   ├── hybrid_retriever.py          # Stage 1 BM25 + dense bi-encoder retriever
│   ├── legal_preemption_evaluator.py# End-to-end full document audit pipeline
│   └── ordinance_preprocessor.py    # Sectional decomposition parser
├── requirements.txt                 # Project dependencies
├── run_demo.py                      # Prototype launcher script
└── README.md                        # Project documentation
```

---

## Quickstart Guide

### 1. Environment Setup

Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/yyaahhzxc/Dulce-Odin-OCDS.git
cd Dulce-Odin-OCDS
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Interactive Web Prototype

Launch the offline prototype demo server:

```bash
python run_demo.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

The prototype provides:
- **Instant Preset Audit:** Test pre-loaded benchmark cases (e.g., Mosqueda aerial spraying ban, GSIS tax exemption, billboard regulations).
- **Custom Section Audit:** Input any draft municipal section text to run real-time hybrid retrieval and NLI screening.
- **Full Document PDF/Text Audit:** Upload multi-section draft ordinances with live Server-Sent Events (SSE) streaming progress.
- **Explainable AI (XAI) Cards:** Interactive review cards displaying contradiction confidence, attention saliency, and direct Lawphil reference links.

### 3. Rebuild the Unified Dual Corpus (Optional)

To compile the master unified dual-corpus file (`data/unified_dual_statutory_provisions.jsonl`) from the constituent national laws and local ordinances:

```bash
python scripts/corpus_build/build_unified_dual_corpus_provisions.py
```

### 4. Run Benchmark Evaluations

To execute the Stage 1 candidate retrieval benchmark:
```bash
python scripts/benchmarks/benchmark_stage1_retrieval.py
```

To execute the Stage 2 cross-encoder NLI screening sweep:
```bash
python scripts/benchmarks/run_stage2_screening.py
```

To run the Tier 2 open-corpus full-document triage evaluation:
```bash
python scripts/benchmarks/run_tier2_system_evaluation.py
```

---

## Experimental Benchmarks Summary

Evaluated on the canonical edge LGU testbed (Intel Core i3-10105F, 8 GB RAM, Windows 11):

| Pipeline Stage / Evaluation Suite | Primary Metric | Baseline | Champion Configuration | Relative Gain / Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1 Retrieval (k = 20)** | Recall@20 | 0.6286 (Sparse BM25) | **0.8686** (Hybrid alpha = 0.40) | +24.00 percentage points |
| **Stage 1 Retrieval (k = 20)** | MRR@20 | 0.4042 (Sparse BM25) | **0.6726** (Hybrid alpha = 0.40) | +26.84 percentage points |
| **Stage 2 NLI Classification** | Accuracy | 33.33% (Chance) | **86.44% +/- 6.05%** (5-Fold CV) | +53.11 percentage points |
| **Stage 2 NLI Classification** | Conflict F1 | 0.3333 (Chance) | **0.8945** (tau* = 0.45) | Champion cross-encoder |
| **Tier 3 Precedent Suite** | Precedent Alignment | N/A | **10 / 11 Precedents (90.9%)** | Exact doctrinal fidelity |
| **Edge Hardware Footprint** | Peak RAM / Latency | N/A | **1,416.4 MB RAM / 12.69 s** | Fully deployable on standard LGU PCs |

---

## Authors & Acknowledgments

- **Ralph Paolo Dulce** (`rp.dulce@addu.edu.ph`) — Proponent, Ateneo de Davao University
- **Yahyah Odin** (`y.odin@addu.edu.ph`) — Proponent, Ateneo de Davao University
- **Mr. Adrian "Ogs" Ablazo** — Thesis Adviser
- **Ma'am Grace Tacadao** — Course Professor, Department of Computer Science

We gratefully acknowledge the **Sangguniang Panlungsod of Davao City** for archival ordinance access and legislative research collaboration.

---

## License

This project is developed for academic research purposes at Ateneo de Davao University. All Philippine statutes and Supreme Court decisions are public domain under Section 176 of Republic Act No. 8293 (Intellectual Property Code of the Philippines).
