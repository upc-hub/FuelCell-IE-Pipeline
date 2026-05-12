"""
scrape_rsc.py
─────────────────────────────────────────────────────────────────────────────
Standalone RSC article scraper for fuel cell literature.

Extracted from the full NLP pipeline (app_staging.py) for public release.
Uses ChemdataExtractor (custom fork: cde_n) for RSC search scraping and
structured article parsing (abstract, results, conclusion sections).

Usage
─────
# 1. Find how many result pages exist for a query:
    python scrape_rsc.py pages --query "ORR catalyst fuel cell"

# 2. Scrape article metadata (DOI, title, URL) from a single page:
    python scrape_rsc.py search --query "ORR catalyst fuel cell" --page 1

# 3. Scrape a range of pages:
    python scrape_rsc.py search --query "ORR catalyst fuel cell" --page 1-5

# 4. Download article HTML by article index from search results:
    python scrape_rsc.py download --query "ORR catalyst fuel cell" --page 1 --article 1

# 5. Full pipeline (search → download → extract structured text):
    python scrape_rsc.py full --query "ORR catalyst fuel cell" --page 1 --article 2 --output ./articles/

Requirements
────────────
- cde_n (custom ChemdataExtractor fork — see README)
- selenium + ChromeDriver (used by both this script and cde_n internally)
- requests

See README.md for full setup instructions.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations  # enables modern type hints on Python 3.7+

import argparse
import json
import os
import re
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple, Union

import requests
from requests.exceptions import ConnectionError, Timeout

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# CDE custom fork — must be cloned/installed from the provided cde_n directory
from cde_n.chemdataextractor.scrape.pub.rsc import RscSearchScraper
from cde_n.chemdataextractor import Document


# ─── Constants ───────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

CONCLUSION_PATTERN = (
    r'\b(Conclusions?|Conclusion(s)? and Outlook|Outlook|Summary|'
    r'Final Remarks|Final Thoughts|Closing Remarks|Concluding Remarks|'
    r'Discussion and Conclusion(s)?|Ending Summary|'
    r'Conclusion(s)? and Future Work|Discussion|Final Discussion)\b'
)
RESULTS_PATTERN = (
    r'\b(Results?|Discussions?|Results? and Discussions?|Results? & Discussions?)\b'
)


# ─── Text utility ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize special characters (matches app_staging.py behaviour)."""
    return re.sub(r'[∼~]', '~', text)


# ─── Section splitting (ported from staging/split_sections_de.py) ─────────────

def split_sections(elements: list) -> tuple:
    """
    Extract structured sections from CDE's parsed document elements.

    Ported directly from staging/split_sections_de.py.

    Returns:
        (abstract_split, conclusion_split, title, publisher, date, doi,
         conclusion_para, result_split, result_para, html_url, figures)
    """
    abstract_split, conclusion_split = [], []
    conclusion_para, result_split, result_para, figures = [], [], [], []
    title, publisher, date, doi, html_url = '', '', '', '', ''

    in_conclusion_section = False
    in_result_section = False

    values = list(elements[0].values())
    try:
        title     = values[0]['title']
        publisher = values[0]['publisher']
        doi       = values[0]['doi']
        date      = values[0]['date']
        html_url  = values[0]['html_url']
    except Exception:
        print("  [Warning] No metadata found in this document.")
        date     = "uncertain date"
        html_url = "uncertain html url"

    r_count = 0
    for i, element in enumerate(elements):
        if not (isinstance(element, dict) and 'type' in element):
            continue
        try:
            if i + 1 >= len(elements):
                continue
            next_element = elements[i + 1]

            # ── Abstract ──────────────────────────────────────────────────────
            if element['type'] == "Heading" and element['content'] == "Abstract":
                abstract = next_element['content']
                abstract_split.append(element['content'])
                abstract_split.append(abstract)

            # ── Figures ───────────────────────────────────────────────────────
            elif element['type'] == "Figure":
                fg_caption = element['caption']['content']
                figures.append(fg_caption)

            # ── Results & Discussion ──────────────────────────────────────────
            elif (element['type'] == "Heading"
                  and re.search(RESULTS_PATTERN, element['content'], re.IGNORECASE)
                  and r_count == 0):
                in_result_section = True
                result_split.append(element['content'])
                heading = element['content']
                r_count = 1

                if '.' not in heading:
                    for j in range(i + 1, len(elements)):
                        subsequent_element = elements[j]
                        if subsequent_element['type'] == "Heading":
                            if re.search(CONCLUSION_PATTERN,
                                         subsequent_element['content'], re.IGNORECASE):
                                in_result_section = False
                                break
                        if subsequent_element['type'] == "Paragraph":
                            result_split.append(subsequent_element['content'])
                            result_para.append(subsequent_element['content'])
                else:
                    head_no = heading.split('.')[0]
                    for j in range(i + 1, len(elements)):
                        subsequent_element = elements[j]
                        if subsequent_element['type'] == "Heading":
                            head_no_1 = subsequent_element['content'].split('.')[0]
                            if head_no_1 != head_no:
                                in_result_section = False
                                break
                        if subsequent_element['type'] == "Paragraph":
                            result_split.append(subsequent_element['content'])
                            result_para.append(subsequent_element['content'])

            # ── Conclusion ────────────────────────────────────────────────────
            elif (element['type'] == "Heading"
                  and re.search(CONCLUSION_PATTERN, element['content'], re.IGNORECASE)):
                in_conclusion_section = True
                conclusion_split.append(element['content'])
                for j in range(i + 1, len(elements)):
                    subsequent_element = elements[j]
                    if subsequent_element['type'] == "Heading":
                        in_conclusion_section = False
                        break
                    if subsequent_element['type'] == "Paragraph":
                        conclusion_split.append(subsequent_element['content'])
                        conclusion_para.append(subsequent_element['content'])

        except IndexError:
            pass

    return (abstract_split, conclusion_split, title, publisher, date, doi,
            conclusion_para, result_split, result_para, html_url, figures)


# ─── Text file generation (ported from staging/generate_txt_html_file_de.py) ──

def generate_text(txt_file_path: str, title: str, publisher: str, date: str,
                  doi: str, abstract: list, result_para: list,
                  conclusion_para: list) -> None:
    """
    Write structured article text to file.

    Ported directly from staging/generate_txt_html_file_de.py.
    Output format:
        Title / Publisher / Date / DOI
        Abstract
        Results & Discussion (one paragraph per entry)
        Conclusion (one paragraph per entry)
    """
    with open(txt_file_path, 'w', encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"Publisher: {publisher}\n")
        f.write(f"Date: {date}\n")
        f.write(f"DOI: {doi}\n\n")
        if len(abstract) > 1:
            f.write(f"Abstract:\n{normalize_text(abstract[1])}\n\n")
        else:
            f.write("Abstract:\n[Not found]\n\n")
        if result_para:
            f.write("Results & Discussion:\n")
            clean_results = [normalize_text(r) for r in result_para if r.strip()]
            f.write("\n".join(clean_results) + "\n\n")
        if conclusion_para:
            f.write("Conclusion:\n")
            clean_conclusion = [normalize_text(c) for c in conclusion_para if c.strip()]
            f.write("\n".join(clean_conclusion) + "\n")


# ─── Core scraping functions ──────────────────────────────────────────────────

def get_pages(query: str, chromedriver_path: Optional[str] = None) -> Tuple[int, int]:
    """
    Query RSC and return total pages and estimated article count instantly.

    Uses Selenium to read the pagination element — one fast request.

    Returns:
        (total_results, total_pages) — e.g. (650, 26)
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--dns-prefetch-disable")

    if not chromedriver_path:
        candidates = [
            "/home/hein/Downloads/chromedriver-linux64/chromedriver-linux64/chromedriver",
            "/usr/local/bin/chromedriver",
            "/usr/bin/chromedriver",
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                chromedriver_path = candidate
                break

    if chromedriver_path:
        driver = webdriver.Chrome(executable_path=chromedriver_path, options=chrome_options)
    else:
        driver = webdriver.Chrome(options=chrome_options)

    encoded_query = urllib.parse.quote(query)
    url = (
        "http://pubs.rsc.org/en/results/journals"
        f"?Category=Journal&AllText={encoded_query}"
        "&ArticleType=Paper&DateRange=true&SelectDate=true"
        "&DateToYear=2024&DateFromYear=2010"
        "&DateFromMonth=01&DateToMonth=06"
        "&PriceCode=False&OpenAccess=true"
    )

    try:
        driver.get(url)
        WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, ".capsule.capsule--article"))
        )
        pagination_elements = driver.find_elements(By.CSS_SELECTOR, "a[class^=paging__btn]")
        if pagination_elements:
            last_page_element = pagination_elements[-1]
            total_pages_text = last_page_element.get_attribute("aria-label")
            last_number = int(re.findall(r"\d+", total_pages_text)[-1])
            total_results = 25 * last_number
            return total_results, last_number
        else:
            articles = driver.find_elements(By.CSS_SELECTOR, ".capsule.capsule--article")
            return len(articles), 1
    finally:
        driver.quit()


def get_doi_titles_and_url(
    query: str,
    page: Union[int, str],
    batch: bool = False
) -> List[Dict]:
    """
    Use ChemdataExtractor's RscSearchScraper to retrieve article metadata.

    Returns:
        List of dicts: [{"doi": ..., "title": ..., "url": ...}, ...]
    """
    articles = []
    encoded_query = urllib.parse.quote(query)

    if not batch:
        scrape = RscSearchScraper().run(encoded_query, int(page))
        if scrape is not None:
            try:
                for entry in scrape.serialize():
                    try:
                        articles.append({
                            "doi":   entry["doi"],
                            "title": entry["title"],
                            "url":   entry["html_url"],
                        })
                    except KeyError:
                        print("  [Warning] Missing doi/title/url for an entry, skipping.")
            except AttributeError:
                pass
    else:
        first, second = map(int, page.split("-"))
        for iteration in range(first, second + 1):
            print(f"  Scraping page {iteration}...")
            scrape = RscSearchScraper().run(encoded_query, iteration)
            if scrape is not None:
                try:
                    for entry in scrape.serialize():
                        try:
                            articles.append({
                                "doi":   entry["doi"],
                                "title": entry["title"],
                                "url":   entry["html_url"],
                            })
                        except KeyError:
                            print("  [Warning] Missing doi/title/url for an entry, skipping.")
                except AttributeError:
                    pass
            time.sleep(1)

    return articles


def download_webpage(
    url: str,
    output_path: str,
    headers: Dict = HEADERS,
    retries: int = 3,
    timeout: int = 10
) -> bool:
    """Download a webpage HTML to a file."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                os.makedirs(
                    os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                    exist_ok=True
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"  Saved: {output_path}")
                return True
            else:
                print(f"  [Attempt {attempt}] HTTP {response.status_code} for {url}")
        except (ConnectionError, Timeout) as e:
            print(f"  [Attempt {attempt}] Connection error: {e}")
    print(f"  [Failed] Could not download {url} after {retries} attempts.")
    return False


def extract_sections_to_txt(html_path: str, txt_path: str) -> bool:
    """
    Parse a downloaded RSC article HTML with CDE and write a structured
    txt file containing: title, publisher, date, DOI, abstract,
    results & discussion, and conclusion sections.

    This replicates the app_staging.py pipeline:
        Document.from_file() → to_json() → split_sections() → generate_text()

    Returns:
        True if successful, False otherwise.
    """
    try:
        with open(html_path, "rb") as f:
            doc = Document.from_file(f)
        data = json.loads(doc.to_json())
    except Exception as e:
        print(f"  [Error] CDE failed to parse HTML: {e}")
        return False

    if 'elements' not in data:
        print("  [Error] No 'elements' found in CDE output.")
        return False

    (abstract, conclusion, title, publisher, date, doi,
     conclusion_para, result_split, result_para, html_url, figures) = split_sections(data['elements'])

    # Report what was found
    print(f"  Title     : {title or '[not found]'}")
    print(f"  DOI       : {doi or '[not found]'}")
    print(f"  Abstract  : {'found' if len(abstract) > 1 else 'NOT FOUND'}")
    print(f"  Results   : {len(result_para)} paragraph(s)")
    print(f"  Conclusion: {len(conclusion_para)} paragraph(s)")

    generate_text(txt_path, title, publisher, date, doi,
                  abstract, result_para, conclusion_para)
    return True


# ─── CLI commands ─────────────────────────────────────────────────────────────

def cmd_pages(args):
    print(f"Searching RSC for: '{args.query}'")
    total_results, total_pages = get_pages(args.query, args.chromedriver)
    print(f"  Total pages   : {total_pages}")
    print(f"  Total articles: ~{total_results} ({total_pages} pages x 25 articles/page)")
    print(f"  Tip: use --page 1 to {total_pages} with the 'search' command.")


def cmd_search(args):
    batch = "-" in str(args.page)
    print(f"Scraping RSC page(s) '{args.page}' for: '{args.query}'")
    articles = get_doi_titles_and_url(args.query, args.page, batch=batch)
    print(f"  Found {len(articles)} articles.")
    for i, art in enumerate(articles, 1):
        print(f"  [{i}] {art['title']}")
        print(f"       DOI: {art['doi']}")
        print(f"       URL: {art['url']}")
    if args.output:
        out_path = (args.output if args.output.endswith(".json")
                    else os.path.join(args.output, "results.json"))
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                    exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {out_path}")


def cmd_download(args):
    batch = "-" in str(args.page)
    print(f"Scraping RSC page(s) '{args.page}' for: '{args.query}'")
    articles = get_doi_titles_and_url(args.query, args.page, batch=batch)
    print(f"  Found {len(articles)} articles.")

    idx = int(args.article) - 1
    if idx < 0 or idx >= len(articles):
        print(f"  [Error] Article index {args.article} is out of range (1–{len(articles)}).")
        return

    article = articles[idx]
    doi_stem = article["doi"].split("/")[-1]
    output_dir = args.output or "./articles"
    output_path = os.path.join(output_dir, doi_stem + ".html")
    print(f"  Downloading article {args.article}: {article['title']}")
    download_webpage(article["url"], output_path)


def cmd_full(args):
    """Full pipeline: search → download HTML → extract structured text via CDE."""
    batch = "-" in str(args.page)
    output_dir = args.output or "./articles"

    # Step 1: Search
    print(f"\n[Step 1] Searching RSC for: '{args.query}', page(s): {args.page}")
    articles = get_doi_titles_and_url(args.query, args.page, batch=batch)
    print(f"  Found {len(articles)} articles.")

    meta_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved to: {meta_path}")

    # Step 2: Download HTML
    idx = int(args.article) - 1
    if idx < 0 or idx >= len(articles):
        print(f"  [Error] Article index {args.article} out of range (1–{len(articles)}).")
        return

    article  = articles[idx]
    doi_stem = article["doi"].split("/")[-1]
    html_path = os.path.join(output_dir, doi_stem + ".html")
    txt_path  = os.path.join(output_dir, doi_stem + ".txt")
    json_path = os.path.join(output_dir, doi_stem + ".json")

    print(f"\n[Step 2] Downloading article {args.article}: {article['title']}")
    success = download_webpage(article["url"], html_path)
    if not success:
        return

    # Step 3: Parse with CDE → structured JSON
    print(f"\n[Step 3] Parsing article with CDE...")
    try:
        with open(html_path, "rb") as f:
            doc = Document.from_file(f)
        cde_json = doc.to_json()
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(cde_json)
        print(f"  CDE JSON saved to: {json_path}")
    except Exception as e:
        print(f"  [Error] CDE parsing failed: {e}")
        return

    # Step 4: Extract sections → clean txt
    print(f"\n[Step 4] Extracting structured sections...")
    ok = extract_sections_to_txt(html_path, txt_path)
    if ok:
        print(f"  Structured text saved to: {txt_path}")
        print(f"\n  ✓ Done. Use {txt_path} as input to Part 2 (DyGIE++ prediction).")
    else:
        print(f"  [Warning] Section extraction failed. Check {json_path} for raw CDE output.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RSC article scraper for fuel cell literature.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--chromedriver", default=None,
        help="Path to ChromeDriver binary (auto-detected if not given)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_pages = subparsers.add_parser("pages", help="Find total result pages for a query.")
    p_pages.add_argument("--query", required=True, help="Search query string.")
    p_pages.set_defaults(func=cmd_pages)

    p_search = subparsers.add_parser("search", help="Scrape article metadata from RSC.")
    p_search.add_argument("--query",  required=True, help="Search query string.")
    p_search.add_argument("--page",   required=True, help="Page number (e.g. 1) or range (e.g. 1-5).")
    p_search.add_argument("--output", default=None,  help="Output directory or JSON file path.")
    p_search.set_defaults(func=cmd_search)

    p_dl = subparsers.add_parser("download", help="Search and download one article HTML.")
    p_dl.add_argument("--query",   required=True, help="Search query string.")
    p_dl.add_argument("--page",    required=True, help="Page number or range.")
    p_dl.add_argument("--article", required=True, help="Article index (1-based) to download.")
    p_dl.add_argument("--output",  default="./articles", help="Output directory.")
    p_dl.set_defaults(func=cmd_download)

    p_full = subparsers.add_parser("full", help="Full pipeline: search → download → extract structured text.")
    p_full.add_argument("--query",   required=True, help="Search query string.")
    p_full.add_argument("--page",    required=True, help="Page number or range.")
    p_full.add_argument("--article", required=True, help="Article index (1-based) to download.")
    p_full.add_argument("--output",  default="./articles", help="Output directory.")
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
