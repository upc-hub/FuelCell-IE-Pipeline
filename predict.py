"""
predict.py
─────────────────────────────────────────────────────────────────────────────
Part 2 of the FuelCell-IE-Pipeline.

Runs the trained DyGIE++ NER/RE model on a structured article text file
(produced by scrape_rsc.py Part 1) and outputs recognized entities and
relations as JSON.

The model is downloaded automatically from Hugging Face on first run.
Requires the DyGIE++ repository to be cloned (for the 'dygie' package).
See README for setup instructions.

Usage
─────
    python predict.py --input articles/d3im00081h.txt --output results/

    # With GPU (faster):
    python predict.py --input articles/d3im00081h.txt --output results/ --cuda 0

    # Using a locally downloaded model:
    python predict.py --input articles/d3im00081h.txt --output results/ \
                      --model-path /path/to/model.tar.gz

Requirements
────────────
    conda activate dygiepp
    (allennlp==1.1.0, allennlp-models==1.1.0, scispacy, en_core_sci_lg)

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

# ── Hugging Face model info ───────────────────────────────────────────────────
HF_REPO_ID  = "UPC-HUB/fuelcell-ner-re"
HF_FILENAME = "model.tar.gz"

# ── Dataset name used during training ────────────────────────────────────────
# Matches the prefix in vocabulary/fuelcell__ner_labels.txt
DATASET_NAME = "fuelcell"

# ── DyGIE++ repo directory ────────────────────────────────────────────────────
# The 'dygie' package lives inside the DyGIE++ repo.
# Set via environment variable DYGIEPP_DIR or auto-detected if repo is cloned
# in the same directory as this script.
DYGIEPP_DIR = os.environ.get(
    "DYGIEPP_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dygiepp")
)


# ─── Step 1: Get model path ───────────────────────────────────────────────────

def get_model_path(model_path_arg: Optional[str]) -> str:
    """Return path to model.tar.gz — local or downloaded from HF."""
    if model_path_arg:
        if not os.path.exists(model_path_arg):
            print(f"[Error] Model not found at: {model_path_arg}")
            sys.exit(1)
        print(f"  Using local model: {model_path_arg}")
        return model_path_arg

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[Error] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"  Downloading from HF: {HF_REPO_ID}")
    print(f"  (cached at {cache_dir} after first run)")
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    except Exception as e:
        print(f"\n[Error] Could not download model: {e}")
        print("\n  If this machine has no internet access, copy the model manually:")
        print(f"    scp htcat:~/.cache/huggingface/hub/models--UPC-HUB--fuelcell-ner-re/snapshots/*/model.tar.gz \\")
        print(f"        ~/dygiepp/model.tar.gz")
        print(f"\n  Then run with --model-path:")
        print(f"    python predict.py --input article.txt --output results/ --model-path ~/dygiepp/model.tar.gz")
        sys.exit(1)
    print(f"  Model ready: {path}")
    return path


# ─── Step 2: Convert article .txt to DyGIE++ input format ────────────────────

def txt_to_dygiepp_input(txt_path: str, output_jsonl: str) -> int:
    """
    Convert structured article text (from scrape_rsc.py) to DyGIE++ format.

    Uses scispaCy for sentence splitting — same tokenizer used at training time.
    Returns number of sentences.
    """
    # Load spaCy model — prefer scientific model, fall back to small English
    import spacy
    try:
        nlp = spacy.load("en_core_sci_lg")
        print("  Tokeniser: en_core_sci_lg")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("  [Warning] en_core_sci_lg not found, using en_core_web_sm.")
            print("  For best results install en_core_sci_lg — see README Step 5.")
        except OSError:
            print("[Error] No spaCy model found. See README Step 5.")
            sys.exit(1)

    # Read article text
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Strip section headers written by scrape_rsc.py generate_text()
    # Keep only the content lines
    section_headers = {
        "Abstract:", "Results & Discussion:", "Conclusion:", "[Not found]"
    }
    content_lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in section_headers:
            continue
        # Strip metadata label prefixes but keep the value
        for prefix in ("Title: ", "Publisher: ", "Date: ", "DOI: "):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        content_lines.append(stripped)

    full_text = " ".join(content_lines)
    doc_key   = os.path.splitext(os.path.basename(txt_path))[0]

    # Sentence split and tokenise
    doc = nlp(full_text)
    sentences = [
        [t.text for t in sent if t.text.strip()]
        for sent in doc.sents
    ]
    sentences = [s for s in sentences if s]  # remove empty

    if not sentences:
        print(f"[Error] No sentences found in {txt_path}")
        sys.exit(1)

    dygiepp_doc = {
        "doc_key":   doc_key,
        "dataset":   DATASET_NAME,
        "sentences": sentences,
    }

    with open(output_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(dygiepp_doc) + "\n")

    print(f"  Document  : {doc_key}")
    print(f"  Sentences : {len(sentences)}")
    return len(sentences)


# ─── Step 3: Run AllenNLP predict ─────────────────────────────────────────────

def run_allennlp_predict(
    model_path: str,
    input_jsonl: str,
    output_jsonl: str,
    cuda_device: int = -1,
) -> None:
    """
    Run AllenNLP predict using the DyGIE++ repo for the 'dygie' package.

    The 'dygie' package is not installable via pip — it lives inside the
    DyGIE++ repository. We run the command from inside that directory so
    Python can find it, or set DYGIEPP_DIR environment variable.
    """
    if not os.path.isdir(DYGIEPP_DIR):
        print(f"[Error] DyGIE++ directory not found at: {DYGIEPP_DIR}")
        print("  Clone it with:")
        print("    git clone https://github.com/dwadden/dygiepp.git")
        print("  Or set the DYGIEPP_DIR environment variable:")
        print("    DYGIEPP_DIR=/path/to/dygiepp python predict.py ...")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "allennlp", "predict",
        model_path,
        input_jsonl,
        "--predictor",       "dygie",
        "--include-package", "dygie",
        "--use-dataset-reader",
        "--output-file",     output_jsonl,
        "--cuda-device",     str(cuda_device),
        "--silent",
    ]
    print(f"  Command: allennlp predict ... (cuda={cuda_device})")
    print(f"  DyGIE++ dir: {DYGIEPP_DIR}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=DYGIEPP_DIR)

    if result.returncode != 0:
        print("[Error] AllenNLP predict failed:")
        # Show last 3000 chars of stderr for diagnosis
        print(result.stderr[-3000:] if result.stderr else "(no stderr output)")
        sys.exit(1)

    if not os.path.exists(output_jsonl) or os.path.getsize(output_jsonl) == 0:
        print("[Error] Output file is empty — prediction failed silently.")
        print(result.stderr[-1000:] if result.stderr else "")
        sys.exit(1)

    print("  AllenNLP predict finished.")


# ─── Step 4: Parse raw output into clean JSON ─────────────────────────────────

def parse_predictions(raw_jsonl: str, output_json: str) -> List[Dict]:
    """
    Convert raw DyGIE++ JSONL output into clean, readable JSON.

    DyGIE++ token indices are global (across all sentences in the doc).
    We convert them to character offsets (start/end) matching brat annotation
    format — i.e. character position of the first and last character of the span
    within the sentence text.
    """
    results = []

    with open(raw_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            sentences = doc.get("sentences", [])
            ner_preds = doc.get("predicted_ner",       [[] for _ in sentences])
            rel_preds = doc.get("predicted_relations", [[] for _ in sentences])

            doc_result = {
                "doc_key":   doc.get("doc_key", "unknown"),
                "entities":  [],
                "relations": [],
                "sentences": [],   # filled after processing all sentences
            }

            token_offset = 0  # cumulative token count across sentences

            for sent_idx, sent_tokens in enumerate(sentences):
                sent_ner  = ner_preds[sent_idx] if sent_idx < len(ner_preds)  else []
                sent_rels = rel_preds[sent_idx] if sent_idx < len(rel_preds)  else []

                # Compute character offset of each token within the sentence
                # Tokens are joined by single spaces (as written by txt_to_dygiepp_input)
                sent_char_offsets = []
                pos = 0
                for tok in sent_tokens:
                    sent_char_offsets.append(pos)
                    pos += len(tok) + 1  # +1 for the space between tokens

                # ── Entities ──────────────────────────────────────────────────
                for span in sent_ner:
                    g_start, g_end, label = span[0], span[1], span[2]
                    # Convert global token indices to sentence-local
                    l_start = max(0, min(g_start - token_offset, len(sent_tokens) - 1))
                    l_end   = max(0, min(g_end   - token_offset, len(sent_tokens) - 1))
                    span_text = " ".join(sent_tokens[l_start : l_end + 1])
                    # Compute character offsets within the sentence
                    # (matches brat annotation format: Z position to T position)
                    char_start = sent_char_offsets[l_start]
                    char_end   = sent_char_offsets[l_end] + len(sent_tokens[l_end]) - 1
                    doc_result["entities"].append({
                        "text":         span_text,
                        "label":        label,
                        "sentence_idx": sent_idx,
                        "char_start":   char_start,
                        "char_end":     char_end,
                    })

                # ── Relations ─────────────────────────────────────────────────
                for rel in sent_rels:
                    s_start, s_end = rel[0], rel[1]
                    o_start, o_end = rel[2], rel[3]
                    rel_label      = rel[4]

                    sl_s = max(0, min(s_start - token_offset, len(sent_tokens) - 1))
                    sl_e = max(0, min(s_end   - token_offset, len(sent_tokens) - 1))
                    ol_s = max(0, min(o_start - token_offset, len(sent_tokens) - 1))
                    ol_e = max(0, min(o_end   - token_offset, len(sent_tokens) - 1))

                    subj_text = " ".join(sent_tokens[sl_s : sl_e + 1])
                    obj_text  = " ".join(sent_tokens[ol_s : ol_e + 1])

                    doc_result["relations"].append({
                        "subject":      subj_text,
                        "relation":     rel_label,
                        "object":       obj_text,
                        "sentence_idx": sent_idx,
                    })

                token_offset += len(sent_tokens)

            # Add sentences for reference (indexed by sentence_idx)
            doc_result["sentences"] = [
                {"sentence_idx": i, "text": " ".join(tokens), "tokens": tokens}
                for i, tokens in enumerate(sentences)
            ]

            results.append(doc_result)

    # Write clean output JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    for doc in results:
        n_ent = len(doc["entities"])
        n_rel = len(doc["relations"])
        print(f"  Entities found  : {n_ent}")
        print(f"  Relations found : {n_rel}")

        if n_ent > 0:
            print("\n  Sample entities (first 5):")
            for e in doc["entities"][:5]:
                print(f"    [{e['label']:20s}] {e['text']}")

        if n_rel > 0:
            print("\n  Sample relations (first 5):")
            for r in doc["relations"][:5]:
                print(f"    {r['subject']!r:30s} --[{r['relation']}]--> {r['object']!r}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run DyGIE++ NER/RE on a fuel cell article (Part 2 of pipeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to article .txt file produced by scrape_rsc.py."
    )
    parser.add_argument(
        "--output", default="./results",
        help="Output directory for JSON results (default: ./results)."
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Local path to model.tar.gz. If omitted, auto-downloaded from HF."
    )
    parser.add_argument(
        "--cuda", type=int, default=-1,
        help="GPU device ID for inference. -1 = CPU (default), 0 = first GPU."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[Error] Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    print(f"\n{'='*60}")
    print(f" FuelCell-IE-Pipeline — Part 2: NER/RE Prediction")
    print(f"{'='*60}")

    # Step 1
    print("\n[Step 1] Locating model...")
    model_path = get_model_path(args.model_path)

    # Step 2
    print("\n[Step 2] Preparing input...")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_jsonl = os.path.join(tmpdir, f"{base_name}_input.jsonl")
        raw_jsonl   = os.path.join(tmpdir, f"{base_name}_raw.jsonl")

        txt_to_dygiepp_input(args.input, input_jsonl)

        # Step 3
        print("\n[Step 3] Running model prediction...")
        run_allennlp_predict(
            model_path, input_jsonl, raw_jsonl, cuda_device=args.cuda
        )

        # Step 4
        print("\n[Step 4] Parsing predictions...")
        final_json = os.path.join(args.output, f"{base_name}_entities_relations.json")
        parse_predictions(raw_jsonl, final_json)

    print(f"\n{'='*60}")
    print(f"  Output: {final_json}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()