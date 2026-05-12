# Information Extraction from Literature for ORR Catalyst in Fuel Cell

This repository contains scripts, configurations, and sample data accompanying the paper:

> **"Information Extraction from Literature for ORR Catalyst in Fuel Cell"**
> Hein Htet, Manae Hirano, Amgad Ahmed Ali Ibrahim, Yutaka Sasaki, Ryoji Asahi
> *Computational Materials Science*, 2026

The pipeline has two parts:

| Part | Input | Script | Output |
|------|-------|--------|--------|
| 1 | RSC search query | `scrape_rsc.py` | Structured article text (`.txt`) |
| 2 | Article text (`.txt`) | `predict.py` | Entities & relations (`.json`) |

---

## Repository Structure

```
.
├── README.md
├── environment.yml          # Conda environment (minimal)
├── requirements.txt         # pip alternative
├── scrape_rsc.py            # Part 1: RSC scraper + article text extraction
├── predict.py               # Part 2: DyGIE++ NER/RE prediction (coming soon)
├── configs/
│   └── fuelcell.jsonnet     # AllenNLP training configuration
└── sample_data/
    ├── README.md            # Description of entity/relation types
    ├── brat_sample/         # Sample raw brat annotations (.ann + .txt)
    └── dygiepp_format/      # Pre-converted DyGIE++ JSON format
        ├── train.json
        ├── dev.json
        └── test.json
```

> **Note:** The full annotated corpus cannot be released due to copyright constraints.
> The sample data (8 documents) is sufficient to demonstrate the full pipeline end-to-end.

---

## Part 1: RSC Article Scraper

`scrape_rsc.py` searches the Royal Society of Chemistry (RSC) publication database,
downloads articles, and extracts structured text sections (abstract, results &
discussion, conclusion) using a custom fork of
[ChemDataExtractor](https://github.com/mcs07/ChemDataExtractor).

### Setup

#### Step 1: Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

#### Step 2: Clone the custom ChemDataExtractor fork

This pipeline uses a modified version of ChemDataExtractor. It must be cloned
**inside the project directory** as `cde_n`:

```bash
git clone https://github.com/YOUR_USERNAME/chemdataextractor-fork.git cde_n
```

> **Why a fork?** The original ChemDataExtractor pip package does not include
> the RSC scraper. Our fork modifies `cde_n/chemdataextractor/scrape/pub/rsc.py`
> with three changes to `perform_search()`:
> 1. Switched from Firefox to **headless Chrome** (`--headless`, `--disable-gpu`, `--no-sandbox`)
> 2. Added a **ChromeDriver path** (configured via `scrape_rsc.py` — see Step 6)
> 3. Added **date range and Open Access filters** to the RSC search URL
>    (`DateFromYear=2010`, `DateToYear=2024`, `OpenAccess=true`)
>
> All other CDE code (entity parsers, text processors, document reader) is unchanged.

After cloning, your directory should look like:

```
YOUR_REPO_NAME/
├── scrape_rsc.py
├── cde_n/
│   └── chemdataextractor/
│       ├── scrape/pub/rsc.py   ← modified: headless Chrome + date filter
│       └── ...                 ← everything else unchanged from original CDE
└── ...
```

#### Step 3: Create conda environment

```bash
conda env create -f environment.yml
conda activate dygiepp
```

#### Step 4: Install PyTorch with CUDA 11.7

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

> For **CPU only** (no GPU):
> ```bash
> pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
> ```

#### Step 5: Install spaCy language models

```bash
# Small English model (required by cde_n)
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

# Large scientific model (required for --use-scispacy in DyGIE++ preprocessing)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_lg-0.2.5.tar.gz
```

#### Step 6: Install ChromeDriver

The RSC scraper uses Selenium + ChromeDriver internally to query the RSC website.
The ChromeDriver version must match your installed Google Chrome version.

```bash
# Check your Chrome version first
google-chrome --version
```

Then download the matching ChromeDriver:
- Chrome 114 and below: https://chromedriver.chromium.org/downloads
- Chrome 115 and above: https://googlechromelabs.github.io/chrome-for-testing/

`scrape_rsc.py` auto-detects ChromeDriver at these locations:
```
/home/<user>/Downloads/chromedriver-linux64/chromedriver-linux64/chromedriver
/usr/local/bin/chromedriver
/usr/bin/chromedriver
```

If yours is elsewhere, pass it explicitly with `--chromedriver`:
```bash
python scrape_rsc.py --chromedriver /path/to/chromedriver pages --query "..."
```

---

### Usage

All commands must be run from the project root directory where `cde_n/` is located.

#### `pages` — Find total result pages for a query

Use this first to know how many pages of results exist before scraping.

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

#### `search` — List articles on a page

```bash
# Single page
python scrape_rsc.py search --query "ORR catalyst fuel cell" --page 1

# Range of pages
python scrape_rsc.py search --query "ORR catalyst fuel cell" --page 1-5

# Save article list to JSON
python scrape_rsc.py search --query "ORR catalyst fuel cell" --page 1 --output ./results/
```

Output:
```
Scraping RSC page(s) '1' for: 'ORR catalyst fuel cell'
  Found 25 articles.
  [1] Highly active ZIF-8@CNT composite catalysts...
       DOI: 10.1039/D3IM00081H
       URL: https://pubs.rsc.org/en/content/articlehtml/...
  [2] ...
```

#### `download` — Download one article HTML

```bash
python scrape_rsc.py download --query "ORR catalyst fuel cell" \
    --page 1 --article 2 --output ./articles/
```

Downloads the HTML of article #2 from page 1 to `./articles/d3im00081h.html`.

#### `full` — Complete pipeline (recommended)

```bash
python scrape_rsc.py full --query "ORR catalyst fuel cell" \
    --page 1 --article 2 --output ./articles/
```

Runs all four steps automatically:

```
[Step 1] Search RSC → articles/metadata.json
[Step 2] Download HTML → articles/d3im00081h.html
[Step 3] Parse with CDE → articles/d3im00081h.json
[Step 4] Extract sections → articles/d3im00081h.txt
```

The output `.txt` file contains the structured article text:

```
Title: Highly active ZIF-8@CNT composite catalysts...
Publisher: Royal Society of Chemistry
Date: 2023/10/20
DOI: 10.1039/D3IM00081H

Abstract:
Developing non-precious metal-based inexpensive and highly active...

Results & Discussion:
Fig. 1a shows the crystallographic features of as-prepared ZIF-8@CNT...
The defects in the as-synthesised ZIF-8@CNT catalysts were analysed...
[all paragraphs concatenated]

Conclusion:
Metal-free and transition-metal-doped ZIF-8@CNT catalysts were prepared...
```

This `.txt` file is the input to **Part 2** (DyGIE++ NER/RE prediction).

---

## Part 2: NER/RE Prediction with DyGIE++

`predict.py` runs the trained DyGIE++ model on a structured article `.txt` file
(produced by Part 1) and outputs entities and relations as JSON.

The model is hosted on Hugging Face at
[UPC-HUB/fuelcell-ner-re](https://huggingface.co/UPC-HUB/fuelcell-ner-re)
and uses [MatSci-BERT](https://huggingface.co/UPC-HUB/matscibert-finetuned-squad)
as the underlying language model.

DyGIE++ does **not** need to be installed — only AllenNLP (already in the
`dygiepp` conda environment) is required.

The pipeline uses [DyGIE++](https://github.com/dwadden/dygiepp) for joint
named entity recognition (NER) and relation extraction (RE).

### Setup

The `dygiepp` conda environment (set up in Part 1) already contains everything needed.
No additional installation is required.

### Usage

#### Machines with internet access

```bash
# Model downloads automatically from Hugging Face on first run (~840MB)
python predict.py --input articles/d3im00081h.txt --output results/

# With GPU (recommended — much faster):
python predict.py --input articles/d3im00081h.txt --output results/ --cuda 0
```

#### Machines without internet access (HPC, closed networks)

Copy the model from a machine that has internet access:

```bash
# Step 1: On the internet-connected machine, download the model
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='UPC-HUB/fuelcell-ner-re', filename='model.tar.gz')
"

# Step 2: Copy to the HPC machine
scp ~/.cache/huggingface/hub/models--UPC-HUB--fuelcell-ner-re/snapshots/*/model.tar.gz     hpc_machine:~/model.tar.gz

# Step 3: Extract MatSci-BERT and create a local model config
mkdir -p /tmp/model_local && cd /tmp/model_local
tar -xzf ~/model.tar.gz
cp -r matscibert_weights ~/matscibert_weights

# Fix config to use absolute path
sed -i "s|matscibert_weights|/home/YOUR_USERNAME/matscibert_weights|g" config.json
tar -czf ~/model_local.tar.gz config.json weights.th vocabulary/

# Step 4: Run prediction
python predict.py     --input article.txt     --output results/     --model-path ~/model_local.tar.gz     --cuda 0
```

#### Full pipeline example (Part 1 → Part 2)

```bash
# Part 1: Scrape and extract article text
python scrape_rsc.py full     --query "ORR catalyst fuel cell"     --page 1 --article 2     --output ./articles/

# Part 2: Run NER/RE prediction
python predict.py     --input ./articles/d3im00081h.txt     --output ./results/     --cuda 0
```

#### Output format

The output JSON file contains extracted entities and relations:

```json
[
  {
    "doc_key": "d3im00081h",
    "entities": [
      {"text": "ZIF-8@CNT",           "label": "catalyst",  "sentence_idx": 2},
      {"text": "carbon nanotube (CNT)","label": "support",   "sentence_idx": 2},
      {"text": "pyrolysis",            "label": "process",   "sentence_idx": 2},
      {"text": "900 °C",              "label": "condition", "sentence_idx": 2},
      {"text": "0.847 V",             "label": "value",     "sentence_idx": 8}
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
| [UPC-HUB/fuelcell-ner-re](https://huggingface.co/UPC-HUB/fuelcell-ner-re) | DyGIE++ model weights + MatSci-BERT embedded | 840MB |
| [UPC-HUB/matscibert-finetuned-squad](https://huggingface.co/UPC-HUB/matscibert-finetuned-squad) | MatSci-BERT fine-tuned on SQuAD (standalone) | 418MB |

---

## Training Your Own Model (DyGIE++)

If you want to retrain using the provided sample data:

#### Step 1: Clone DyGIE++

```bash
git clone https://github.com/dwadden/dygiepp.git
cd dygiepp
```

#### Step 2: Convert brat annotations to DyGIE++ format

```bash
python scripts/new-dataset/brat_to_input.py \
    /path/to/brat_annotations/ \
    ./output.jsonl \
    fuelcell \
    --use-scispacy \
    --coref
```

> Each line in `output.jsonl` represents one document (article).

#### Step 3: Split into train / dev / test

Manually copy lines from `output.jsonl` into:
- `train.jsonl` — training documents
- `dev.jsonl`   — validation documents
- `test.jsonl`  — test documents

#### Step 4: Collate (split long documents to avoid out-of-memory)

```bash
python scripts/data/shared/collate.py \
    /path/to/train_dev_test_dir/ \
    /path/to/output_dir/
```

#### Step 5: Rename `.jsonl` → `.json`

```bash
mv output_dir/train.jsonl output_dir/train.json
mv output_dir/dev.jsonl   output_dir/dev.json
mv output_dir/test.jsonl  output_dir/test.json
```

#### Step 6: Train

```bash
bash scripts/train.sh fuelcell
```

#### Step 7: Evaluate

```bash
allennlp evaluate \
    models/fuelcell/model.tar.gz \
    test.json \
    --cuda-device 0 \
    --include-package dygie \
    --output-file models/fuelcell/metrics_test.json
```

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
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods
               in Natural Language Processing},
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

> The full annotated corpus is not released due to RSC copyright restrictions
> on article content. The sample data provided is sufficient to reproduce
> the pipeline end-to-end.
