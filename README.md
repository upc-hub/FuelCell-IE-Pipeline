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
│       ├── scrape/pub/rsc.py   ← modified: headless Chrome + date filter
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

## Requirements

- Linux (Ubuntu 18.04+ recommended) or macOS
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Google Chrome browser
- Internet access (for Part 1 RSC scraping and model download)
- GPU with CUDA 11.7 (optional — CPU works but is slower for Part 2)

---

## Environment Setup

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
├── predict.py
├── cde_n/
│   └── chemdataextractor/
│       └── scrape/pub/rsc.py
└── ...
```

> **What was modified in cde_n?**
> Only `cde_n/chemdataextractor/scrape/pub/rsc.py` — three changes to
> `perform_search()`:
> 1. Switched from Firefox to **headless Chrome**
> 2. Uses **chromedriver-autoinstaller** (no manual driver download needed)
> 3. Added **date range and Open Access filters** (2010–2024) to RSC search URL

### Step 3: Create conda environment

```bash
conda env create -f environment.yml
conda activate fuelcell-ie
```

> If you prefer pip, see [pip installation](#pip-installation-alternative).

### Step 4: Install PyTorch

Install **after** activating the environment:

```bash
# GPU (CUDA 11.7) — recommended for Part 2:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html

# CPU only (works but Part 2 will be slow):
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```

> The warning `allennlp 1.1.0 requires torch<1.7.0` can be safely ignored —
> torch 1.13.1 works correctly with this model.

### Step 5: Install spaCy language models

```bash
# English model — required by cde_n for RSC scraping (Part 1)
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

# Scientific English model — required by predict.py for tokenisation (Part 2)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_lg-0.3.0.tar.gz
```

### Step 6: Verify Chrome is installed

`scrape_rsc.py` uses Selenium + Chrome. ChromeDriver is installed
**automatically** via `chromedriver-autoinstaller` — no manual download needed.

```bash
# Verify Chrome is available
google-chrome --version    # Linux
# or: /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version  # macOS
```

If Chrome is not installed:
```bash
# Ubuntu/Debian
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo apt-get install google-chrome-stable
```

---

## pip Installation Alternative

If conda is not available (use `requirements.txt` instead of `environment.yml`):

```bash
# Python 3.7 required
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html   # GPU
# or: pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1  # CPU

pip install -r requirements.txt

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_lg-0.3.0.tar.gz
```

> **Note:** If you use conda, ignore `requirements.txt` —
> `environment.yml` already covers everything.

---

## Part 1: RSC Article Scraper

`scrape_rsc.py` searches RSC, downloads article HTML, and extracts structured
text (abstract, results & discussion, conclusion) using the `cde_n` fork.

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

Conclusion:
Metal-free and transition-metal-doped ZIF-8@CNT catalysts...
```

This `.txt` file is the input to **Part 2**.

---

## Part 2: NER/RE Prediction

`predict.py` runs the trained DyGIE++ model on article text and outputs
entities and relations as JSON.

The model is hosted on Hugging Face:
[UPC-HUB/fuelcell-ner-re](https://huggingface.co/UPC-HUB/fuelcell-ner-re)

DyGIE++ does **not** need to be installed — only AllenNLP (already in
the `fuelcell-ie` conda environment) is required.

### Basic usage

```bash
# Download model automatically on first run (~840MB) and run on CPU:
python predict.py --input articles/d3im00081h.txt --output results/

# Run on GPU (recommended — much faster):
python predict.py --input articles/d3im00081h.txt --output results/ --cuda 0
```

### First-run model setup

On first run, `predict.py` downloads the model from Hugging Face and caches
it in `~/.cache/huggingface/hub/`. Subsequent runs use the cache directly.

If you want to pre-download the model manually:

```bash
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='UPC-HUB/fuelcell-ner-re', filename='model.tar.gz')
print('Model cached at:', path)
"
```

### Machines without internet access

If your machine has no internet (e.g. HPC GPU nodes), download the model
on an internet-connected machine first, then prepare a local copy:

```bash
# Step 1: On an internet-connected machine — download and prepare
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='UPC-HUB/fuelcell-ner-re', filename='model.tar.gz')
"

mkdir -p /tmp/model_local && cd /tmp/model_local
tar -xzf ~/.cache/huggingface/hub/models--UPC-HUB--fuelcell-ner-re/snapshots/*/model.tar.gz
cp -r matscibert_weights ~/matscibert_weights

# Replace relative path with absolute path for your machine
sed -i "s|matscibert_weights|$HOME/matscibert_weights|g" config.json
tar -czf ~/model_local.tar.gz config.json weights.th vocabulary/

# Step 2: Copy model_local.tar.gz to the no-internet machine
# (via scp, shared filesystem, USB, etc.)

# Step 3: Run on the no-internet machine
python predict.py \
    --input articles/d3im00081h.txt \
    --output results/ \
    --model-path ~/model_local.tar.gz \
    --cuda 0
```

### Full pipeline example (Part 1 → Part 2)

```bash
conda activate fuelcell-ie
cd FuelCell-IE-Pipeline

# Part 1: scrape and extract article text
python scrape_rsc.py full \
    --query "ORR catalyst fuel cell" \
    --page 1 --article 2 \
    --output ./articles/
# → produces: articles/d3im00081h.txt

# Part 2: run NER/RE prediction
python predict.py \
    --input ./articles/d3im00081h.txt \
    --output ./results/
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
> The sample data provided is sufficient to reproduce the pipeline end-to-end.
