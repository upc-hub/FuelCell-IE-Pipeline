# FuelCell-IE-Pipeline

Information extraction pipeline for ORR catalyst literature — scrapes RSC articles and runs NER/RE to extract entities and relations.

> Associated paper: *Information Extraction from Literature for ORR Catalyst in Fuel Cell*, Hein Htet et al., Computational Materials Science, 2026

---

## Overview

| Part | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `scrape_rsc.py` | RSC search query | Structured article text (`.txt`) |
| 2 | `predict.py` | Article text (`.txt`) | Entities & relations (`.json`) |

**Requirements:** Linux or macOS · Python 3.7 · Conda · Google Chrome · Internet access

---

## Setup

### 1. Clone repositories

```bash
git clone https://github.com/upc-hub/FuelCell-IE-Pipeline.git
cd FuelCell-IE-Pipeline

# ChemdataExtractor fork (required for Part 1)
git clone https://github.com/upc-hub/chemdataextractor-fork.git cde_n

# DyGIE++ (required for Part 2)
git clone https://github.com/dwadden/dygiepp.git
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate fuelcell-ie
```

### 3. Install PyTorch

```bash
# CPU (works on any machine):
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# GPU — CUDA 11.7 only:
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
#     -f https://download.pytorch.org/whl/torch_stable.html
```

> The warning `allennlp 1.1.0 requires torch<1.7.0` can be safely ignored.

### 4. Install spaCy models

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_lg-0.3.0.tar.gz
```

### 5. Install ChromeDriver

ChromeDriver is installed automatically — you only need Google Chrome:

```bash
google-chrome --version   # verify Chrome is available
pip install chromedriver-autoinstaller==0.6.4
```

---

## Part 1: RSC Scraper

```bash
# Find how many result pages exist
python scrape_rsc.py pages --query "ORR catalyst fuel cell"

# List articles on a page
python scrape_rsc.py search --query "ORR catalyst fuel cell" --page 1

# Full pipeline: search → download → extract structured text
python scrape_rsc.py full --query "ORR catalyst fuel cell" \
    --page 1 --article 2 --output ./articles/
```

Output `.txt` format:
```
Title: ...
Publisher: Royal Society of Chemistry
Date: ...
DOI: ...

Abstract:
...

Results & Discussion:
...

Conclusion:
...
```

---

## Part 2: NER/RE Prediction

Model: [UPC-HUB/fuelcell-ner-re](https://huggingface.co/UPC-HUB/fuelcell-ner-re) (~840MB, downloads automatically on first run)

```bash
# CPU (model downloads automatically):
python predict.py --input articles/d3im00081h.txt --output results/

# GPU:
python predict.py --input articles/d3im00081h.txt --output results/ --cuda 0

# Local model (no internet):
python predict.py --input articles/d3im00081h.txt --output results/ \
    --model-path ~/model_local.tar.gz
```

### No-internet setup (e.g. HPC GPU nodes)

```bash
# On internet-connected machine:
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='UPC-HUB/fuelcell-ner-re', filename='model.tar.gz')"

mkdir -p /tmp/model_local && cd /tmp/model_local
tar -xzf ~/.cache/huggingface/hub/models--UPC-HUB--fuelcell-ner-re/snapshots/*/model.tar.gz
cp -r matscibert_weights ~/matscibert_weights
sed -i "s|matscibert_weights|$HOME/matscibert_weights|g" config.json
tar -czf ~/model_local.tar.gz config.json weights.th vocabulary/
```

### Output format

```json
[{
  "doc_key": "d3im00081h",
  "entities": [
    {"text": "ZIF-8@CNT",  "label": "catalyst",  "sentence_idx": 2},
    {"text": "pyrolysis",  "label": "process",   "sentence_idx": 2},
    {"text": "0.847 V",    "label": "value",     "sentence_idx": 8}
  ],
  "relations": [
    {"subject": "ZIF-8@CNT", "relation": "related_to", "object": "pyrolysis"}
  ]
}]
```

### Entity & Relation Types

| Entity | Description | Relation | Description |
|--------|-------------|----------|-------------|
| `catalyst` | ORR catalyst | `related_to` | General relationship |
| `support` | Catalyst support | `equivalent` | Abbreviation ↔ full name |
| `additive` | Additive | | |
| `electrolyte` | Electrolyte | | |
| `precursors` | Precursor material | | |
| `other_material` | Other materials | | |
| `material_reference` | Reference material | | |
| `property` | Physical/chemical property | | |
| `structure` | Material structure | | |
| `process` | Synthesis process | | |
| `condition` | Experimental condition | | |
| `value` | Numerical value with unit | | |

---

## Citation

```bibtex
@article{htet2026fuelcell,
  title   = {Information Extraction from Literature for ORR Catalyst in Fuel Cell},
  author  = {Hein Htet and Manae Hirano and Amgad Ahmed Ali Ibrahim
             and Yutaka Sasaki and Ryoji Asahi},
  journal = {Computational Materials Science},
  year    = {2026}
}
```

Please also cite [DyGIE++](https://github.com/dwadden/dygiepp) and [ChemDataExtractor](https://github.com/mcs07/ChemDataExtractor) if you use this pipeline.

---

## License

MIT License
