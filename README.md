# FuelCell-IE-Pipeline

Information extraction pipeline for ORR catalyst literature in fuel cells.

This repository accompanies the paper:

> **Information Extraction from Literature for ORR Catalyst in Fuel Cell**
> Hein Htet, Manae Hirano, Amgad Ahmed Ali Ibrahim, Yutaka Sasaki, Ryoji Asahi
> *Computational Materials Science*, 2026

The pipeline has two parts:

| Part | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `scrape_rsc.py` | RSC search query | Structured article text (`.txt`) |
| 2 | `predict.py` | Article text (`.txt`) | Entities & relations (`.json`) |

---

## Repository Structure

```
FuelCell-IE-Pipeline/
├── README.md
├── environment.yml        # Conda environment for both scripts
├── requirements.txt       # pip alternative
├── scrape_rsc.py          # Part 1: RSC scraper + CDE text extraction
├── predict.py             # Part 2: DyGIE++ NER/RE prediction
├── cde_n/                 # Custom ChemdataExtractor fork (cloned separately)
│   └── chemdataextractor/
│       ├── scrape/pub/rsc.py   ← modified for headless Chrome + date filter
│       └── ...
├── configs/
│   └── fuelcell.jsonnet   # AllenNLP training configuration (reference only)
└── sample_data/
    ├── README.md
    ├── brat_sample/        # 8 sample brat annotations (.ann + .txt)
    └── dygiepp_format/     # Pre-converted DyGIE++ JSON
        ├── train.json
        ├── dev.json
        └── test.json
```

> The full annotated corpus cannot be released due to RSC copyright constraints.
> The 8 sample documents are sufficient to demonstrate the pipeline end-to-end.

---

## Environment Setup

Both `scrape_rsc.py` and `predict.py` run in a **single conda environment**
named `fuelcell-ie`. All setup steps are run on **mercury** unless noted.

> **Machine setup used in this work:**
> - `mercury` — internet access, Chrome installed, NVIDIA T400 (CUDA 11.6, 2GB)
>   → runs Part 1 and Part 2 CPU (testing)
> - `htcatg02` — NVIDIA GPU with CUDA 11.7, shares filesystem with mercury
>   → runs Part 2 GPU (production)
>
> If your setup differs, adjust machine names accordingly.

### Step 1: Clone this repository

```bash
git clone https://github.com/upc-hub/FuelCell-IE-Pipeline.git
cd FuelCell-IE-Pipeline
```

### Step 2: Clone the custom ChemdataExtractor fork

The RSC scraper requires a modified version of ChemdataExtractor.
Clone it **inside the project directory** as `cde_n`:

```bash
git clone https://github.com/upc-hub/chemdataextractor-fork.git cde_n
```

Your directory should now look like:
```
FuelCell-IE-Pipeline/
├── scrape_rsc.py
├── cde_n/
│   └── chemdataextractor/
│       └── scrape/pub/rsc.py
└── ...
```

> **What was modified in cde_n?**
> Only `cde_n/chemdataextractor/scrape/pub/rsc.py` was changed — three modifications
> to `perform_search()`: switched from Firefox to headless Chrome, added ChromeDriver
> path, and added date range + Open Access filters (2010–2024) to the RSC search URL.
> All other CDE code is unchanged from the original.

### Step 3: Create conda environment

```bash
conda env create -f environment.yml
conda activate fuelcell-ie
```

> If you prefer pip over conda, see [pip installation](#pip-installation-alternative) below.

### Step 4: Install PyTorch

Install **after** activating the environment.

```bash
# On mercury (CUDA 11.6 — model requires 11.7, so use CPU version):
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

# On htcatg02 (CUDA 11.7 — GPU inference for Part 2):
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

> Mercury has an NVIDIA T400 (CUDA 11.6, 2GB VRAM). Since the model requires
> CUDA 11.7 and needs more than 2GB VRAM, use CPU on mercury and GPU on htcatg02.

### Step 5: Install spaCy language models

```bash
# English model — required by cde_n for RSC scraping (Part 1)
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

# Scientific English model — required by predict.py for tokenisation (Part 2)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_lg-0.3.0.tar.gz
```

### Step 6: Chrome browser

`scrape_rsc.py` uses Selenium to query the RSC website. ChromeDriver is
**installed automatically** via `chromedriver-autoinstaller` (included in
the conda environment) — no manual driver download needed.

You only need Google Chrome installed on the machine:

```bash
# Verify Chrome is available
google-chrome --version
```

> If you need to use a specific ChromeDriver path, pass it with `--chromedriver`:
> ```bash
> python scrape_rsc.py --chromedriver /path/to/chromedriver pages --query "..."
> ```

---

## pip Installation Alternative

If conda is not available, use `requirements.txt` instead:

```bash
# Python 3.7 required
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_lg-0.3.0.tar.gz
```

> **Note:** If you are using conda (recommended), ignore `requirements.txt` —
> `environment.yml` already covers everything.

---

## Part 1: RSC Article Scraper

`scrape_rsc.py` searches RSC, downloads article HTML, and extracts structured
text sections (abstract, results & discussion, conclusion) using the `cde_n`
ChemdataExtractor fork.

Run all commands from the project root directory where `cde_n/` is located.

### `pages` — find total result pages for a query

```bash
python scrape_rsc.py pages --query "ORR catalyst fuel cell"
```

Output:
```
Searching RSC for: 'ORR catalyst fuel cell'
  Total pages   : 26
  Total articles: ~650 (26 pages x 25 articles/page)
  Tip: use --page 1 to 26 with the 'search' command.
```

### `search` — list articles on a page

```bash
# Single page
python scrape_rsc.py search --query "ORR catalyst fuel cell" --page 1

# Range of pages (saves to JSON)
python scrape_rsc.py search --query "ORR catalyst fuel cell" --page 1-5 \
    --output ./results/
```

### `download` — download one article HTML

```bash
python scrape_rsc.py download --query "ORR catalyst fuel cell" \
    --page 1 --article 2 --output ./articles/
```

### `full` — complete pipeline (recommended)

```bash
python scrape_rsc.py full --query "ORR catalyst fuel cell" \
    --page 1 --article 2 --output ./articles/
```

Runs all steps automatically:
```
[Step 1] Search RSC        → articles/metadata.json
[Step 2] Download HTML     → articles/d3im00081h.html
[Step 3] Parse with CDE    → articles/d3im00081h.json
[Step 4] Extract sections  → articles/d3im00081h.txt
```

The output `.txt` file:
```
Title: Highly active ZIF-8@CNT composite catalysts...
Publisher: Royal Society of Chemistry
Date: 2023/10/20
DOI: 10.1039/D3IM00081H

Abstract:
Developing non-precious metal-based...

Results & Discussion:
Fig. 1a shows the crystallographic features...
[all paragraphs concatenated]

Conclusion:
Metal-free and transition-metal-doped ZIF-8@CNT catalysts...
```

This `.txt` file is the input to **Part 2**.

---

## Part 2: NER/RE Prediction

`predict.py` runs the trained DyGIE++ model on article text and outputs
recognized entities and relations as JSON.

The model is hosted on Hugging Face:
[UPC-HUB/fuelcell-ner-re](https://huggingface.co/UPC-HUB/fuelcell-ner-re)

DyGIE++ does **not** need to be installed — only AllenNLP (already in the
`fuelcell-ie` conda environment) is required.

### Machines with internet access

The model downloads automatically (~840MB) on first run:

```bash
# CPU (mercury — for testing and verification):
python predict.py --input articles/d3im00081h.txt --output results/

# GPU (htcatg02 — for real use, after model_local.tar.gz is prepared):
python predict.py --input articles/d3im00081h.txt --output results/ \
    --model-path ~/model_local.tar.gz --cuda 1
```

### Machines without internet access (shared filesystem)

In HPC environments where GPU nodes have no internet but share a filesystem
with an internet-connected machine, download and prepare the model once —
it is immediately accessible on all nodes.

```bash
# ── On mercury (internet access) ──────────────────────────────────────────────

conda activate fuelcell-ie
cd FuelCell-IE-Pipeline

# Step 1: Download model from Hugging Face (~840MB, once only)
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='UPC-HUB/fuelcell-ner-re', filename='model.tar.gz')
print('Downloaded to:', path)
"

# Step 2: Extract and fix MatSci-BERT path for this machine
mkdir -p /tmp/model_local && cd /tmp/model_local
tar -xzf ~/.cache/huggingface/hub/models--UPC-HUB--fuelcell-ner-re/snapshots/*/model.tar.gz
cp -r matscibert_weights ~/matscibert_weights

sed -i "s|matscibert_weights|/home/$USER/matscibert_weights|g" config.json
tar -czf ~/model_local.tar.gz config.json weights.th vocabulary/
cd ~/FuelCell-IE-Pipeline

# ── On mercury (CPU, no GPU) ───────────────────────────────────────────────────
# Works fine for testing — just slower than GPU

python predict.py \
    --input articles/d3im00081h.txt \
    --output results/
    # --cuda -1 is the default (CPU)

# ── On htcatg02 (GPU, shared filesystem — no copy needed) ─────────────────────

rlogin htcatg02
conda activate fuelcell-ie          # same env, shared filesystem
cd ~/FuelCell-IE-Pipeline

python predict.py \
    --input articles/d3im00081h.txt \
    --output results/ \
    --model-path ~/model_local.tar.gz \
    --cuda 1
```

### Full pipeline example (Part 1 → Part 2)

```bash
# ── On mercury: Part 1 (internet + Chrome available) ─────────────────────────
conda activate fuelcell-ie
cd FuelCell-IE-Pipeline

python scrape_rsc.py full \
    --query "ORR catalyst fuel cell" \
    --page 1 --article 2 \
    --output ./articles/
# → produces: articles/d3im00081h.txt

# ── On mercury: Part 2 CPU (for testing, no GPU needed) ───────────────────────
python predict.py \
    --input ./articles/d3im00081h.txt \
    --output ./results/ \
    --model-path ~/model_local.tar.gz
# → produces: results/d3im00081h_entities_relations.json

# ── On htcatg02: Part 2 GPU (for faster/batch processing) ─────────────────────
# Shared filesystem — no file copying needed
rlogin htcatg02
conda activate fuelcell-ie
cd ~/FuelCell-IE-Pipeline

python predict.py \
    --input ./articles/d3im00081h.txt \
    --output ./results/ \
    --model-path ~/model_local.tar.gz \
    --cuda 1
# → produces: results/d3im00081h_entities_relations.json
```

### Output format

```json
[
  {
    "doc_key": "d3im00081h",
    "entities": [
      {"text": "ZIF-8@CNT",            "label": "catalyst",  "sentence_idx": 2},
      {"text": "carbon nanotube (CNT)", "label": "support",   "sentence_idx": 2},
      {"text": "pyrolysis",             "label": "process",   "sentence_idx": 2},
      {"text": "900 °C",               "label": "condition", "sentence_idx": 2},
      {"text": "0.847 V",              "label": "value",     "sentence_idx": 8}
    ],
    "relations": [
      {"subject": "ZIF-8@CNT", "relation": "related_to", "object": "carbon nanotube (CNT)"},
      {"subject": "ZIF-8@CNT", "relation": "related_to", "object": "pyrolysis"}
    ]
  }
]
```

### Entity Types

| Type | Description | Example |
|------|-------------|---------|
| `catalyst` | ORR catalyst material | Fe1Co2-ZNT-900 |
| `support` | Catalyst support | carbon nanotube (CNT) |
| `additive` | Additive | KOH |
| `electrolyte` | Electrolyte | Nafion |
| `precursors` | Precursor material | ZIF-8 |
| `other_material` | Other materials | Pt/C |
| `material_reference` | Reference material | commercial Pt/C |
| `property` | Physical/chemical property | half-wave potential |
| `structure` | Material structure | microporous |
| `process` | Synthesis/treatment process | pyrolysis |
| `condition` | Experimental condition | 900 °C |
| `value` | Numerical value with unit | 0.847 V |

### Relation Types

| Type | Description |
|------|-------------|
| `related_to` | General relationship between entities |
| `equivalent` | Material equivalence (e.g. abbreviation ↔ full name) |

### Hugging Face Model Repositories

| Repository | Contents | Size |
|------------|----------|------|
| [UPC-HUB/fuelcell-ner-re](https://huggingface.co/UPC-HUB/fuelcell-ner-re) | DyGIE++ model + MatSci-BERT weights | 840MB |
| [UPC-HUB/matscibert-finetuned-squad](https://huggingface.co/UPC-HUB/matscibert-finetuned-squad) | MatSci-BERT fine-tuned on SQuAD (standalone) | 418MB |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{htet2026fuelcell,
  title   = {Information Extraction from Literature for ORR Catalyst in Fuel Cell},
  author  = {Hein Htet and Manae Hirano and Amgad Ahmed Ali Ibrahim
             and Yutaka Sasaki and Ryoji Asahi},
  journal = {Computational Materials Science},
  year    = {2026}
}
```

Please also cite DyGIE++:

```bibtex
@inproceedings{wadden-etal-2019-entity,
  title     = {Entity, Relation, and Event Extraction with Contextualized Span Representations},
  author    = {Wadden, David and Wennberg, Ulme and Luan, Yi and Hajishirzi, Hannaneh},
  booktitle = {Proceedings of EMNLP-IJCNLP 2019},
  year      = {2019}
}
```

And ChemDataExtractor:

```bibtex
@article{swain2016chemdataextractor,
  title   = {ChemDataExtractor: A Toolkit for Automated Chemical Information
             Extraction from the Scientific Literature},
  author  = {Swain, Matthew C. and Cole, Jacqueline M.},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {56},
  number  = {10},
  pages   = {1894--1904},
  year    = {2016}
}
```

---

## License

Code in this repository is released under the **MIT License**.
Sample data annotations are released under **CC BY 4.0**.

> The full annotated corpus is not released due to RSC copyright restrictions.
> The sample data is sufficient to reproduce the pipeline end-to-end.
