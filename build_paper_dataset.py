#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import arxiv
import json
from pyalex import Works, config
import csv
from pdfminer.high_level import extract_text
import datetime
from datetime import datetime, timezone, timedelta

OPENALEX_API = "https://api.openalex.org/works"
CONTACT_EMAIL = "someone@vt.edu"
MAX_PER_PAGE = 200
config.email = CONTACT_EMAIL


def _id_from_arxiv_url(url):
    return re.sub(r".*/", "", url)


def _extract_arxiv_source_id(doc):
    oa = doc.get("open_access")
    if oa and oa.get("oa_status") == "green":
        url = oa.get("oa_url")
        if "arxiv.org" in url:
            return _id_from_arxiv_url(url)

    return None


def get_ids_from_openalex(publication_year, n: int) -> list[str]:
    ARXIV_SOURCE_ID = 'S4306400194'
    TOPIC = "t10036"  # Advanced Neural Network Applications

    works_q = (
        Works()
        .filter(
            repository=ARXIV_SOURCE_ID,
            primary_topic={"id": TOPIC},
            publication_year=publication_year
        )
        .sort(cited_by_count="desc")
    )

    ids = []
    # paginate through OpenAlex results until we have n arXiv IDs
    for page in works_q.paginate(per_page=MAX_PER_PAGE):
        for doc in page:
            arxiv_id = _extract_arxiv_source_id(doc)
            if arxiv_id:
                ids.append(arxiv_id)

            if len(ids) >= n:
                return ids

    return ids


def get_ids_from_arxiv(publication_year, n: int) -> list[str]:

    # start at Jan 1 of the given year (UTC)
    start = datetime(publication_year, 1, 1, tzinfo=timezone.utc)
    start_str = start.strftime("%Y%m%d%H%M")

    # end is now (UTC)
    # end is yesterday (UTC)
    end_time = datetime.now(timezone.utc) - timedelta(days=1)
    end_str = end_time.strftime("%Y%m%d%H%M")

    # Hardcode the field as "machine learning"
    field_query = "cs.LG"

    field_query = "cat:cs.LG"
    date_filter = f"submittedDate:[{start_str} TO {end_str}]"

    query = f"{field_query} AND {date_filter}"

    search = arxiv.Search(
        query=query,
        max_results=n,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()

    ids = []
    for paper in client.results(search):
        ids.append(_id_from_arxiv_url(paper.entry_id))

    return ids


def fetch_arxiv_paper(arxiv_id: str,
                      download_pdf: bool = False,
                      pdf_dir: str | Path = ".") -> dict:
    """
    Grab basic metadata for one arXiv paper (and optionally save its PDF).
    """
    print("Processing", arxiv_id)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id], max_results=1)

    try:
        result = next(client.results(search))
    except StopIteration:
        raise ValueError(f"No arXiv record found for '{arxiv_id}'")

    pdf_path = None
    if download_pdf:
        pdf_dir = Path(pdf_dir)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{arxiv_id}.pdf"
        result.download_pdf(dirpath=pdf_dir, filename=pdf_path.name)

    return {
        "id": arxiv_id,
        "title": result.title,
        "authors": [a.name for a in result.authors],
        "abstract": result.summary.replace("\n", " ").strip(),
        "published": result.published.isoformat(),
        "pdf_path": str(pdf_path) if pdf_path else None,
    }


def command_list_ids(args):
    if args.source == "openalex":
        ids = get_ids_from_openalex(args.year, args.num)
    elif args.source == "arxiv":
        ids = get_ids_from_arxiv(args.year, args.num)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write IDs as one-per-line CSV without headers
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for paper_id in ids:
            writer.writerow([paper_id])
    print(f"Saved list of IDs to {output_file}")


def command_download_pdfs(args):

    # Read all arXiv IDs from the CSV file (one per line, no header)
    ids = []
    with args.ids.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                ids.append(row[0])

    records = []
    pdf_dir = Path(args.output)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Download PDF and collect metadata
    for paper_id in ids:
        record = fetch_arxiv_paper(
            paper_id, download_pdf=True, pdf_dir=pdf_dir
        )
        records.append(record)

    # Write all records to a JSON file
    output_file = pdf_dir / "papers.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {output_file}")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        text = extract_text(str(pdf_path))
        return "\n".join(line.rstrip() for line in text.splitlines())
    except Exception as e:
        print(f"[!] {pdf_path.name}: {type(e).__name__} – skipped ({e})")

    return None


def command_extract(args):
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    extracted_records = []
    for record in records:
        print("Processing", record["id"])
        pdf_path = Path(record.get("pdf_path"))
        text = extract_text_from_pdf(pdf_path)
        if text is None:
            continue
        record["text"] = text
        extracted_records.append(record)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(extracted_records, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="build arxiv dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command to list arXiv ids.
    parser_ids = subparsers.add_parser(
        "ids", help="Fetch and list arXiv IDs.")
    parser_ids.add_argument(
        "--year",
        required=True,
        type=int,
        help="Publication year for papers.")
    parser_ids.add_argument(
        "--num", type=int, default=MAX_PER_PAGE,
        help="Number of papers to fetch (default: %(default)s)")
    parser_ids.add_argument(
        "--source",
        choices=["openalex", "arxiv"],
        default="openalex",
        help="Source to fetch IDs from: openalex or arxiv.")
    parser_ids.add_argument(
        "--output",
        help="Path to output csv",
        default="ids.csv")

    parser_ids.set_defaults(func=command_list_ids)

    # Command to download pdfs.
    parser_download = subparsers.add_parser(
        "download", help="Download PDFs for given arXiv IDs.")
    parser_download.add_argument(
        "--ids", type=Path, required=True,
        help="Path to CSV file containing arXiv IDs, one per line."
    )
    parser_download.add_argument(
        "--output", default="arxiv_papers",
        help="Directory to save PDFs and metadata."
    )
    parser_download.set_defaults(func=command_download_pdfs)

    parser_extract = subparsers.add_parser(
        "extract",
        help="Extract text from paper PDFs"
    )
    parser_extract.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSON file."
    )
    parser_extract.add_argument(
        "--output",
        type=Path,
        default="./papers-extracted.json",
        help="Path for the output JSON file."
    )
    parser_extract.set_defaults(func=command_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
