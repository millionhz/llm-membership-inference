#!/usr/bin/env python3
import json
import argparse
import csv
from sentence_transformers import SentenceTransformer, util


def compute_similarity(input_path, model_name, output_csv=None):
    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load sentence-transformer model
    model = SentenceTransformer(model_name)

    # Prepare CSV writer if needed
    if output_csv:
        csv_file = open(output_csv, 'w', newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_file, fieldnames=['index', 'similarity'])
        writer.writeheader()

    # Compute and report similarities
    for idx, entry in enumerate(data):
        ground = entry.get('ground', '').strip()
        completion = entry.get('completion', '').strip()

        if not ground or not completion:
            print(
                f"[WARN] Entry {idx} missing ground or completion, skipping.")
            continue

        # Embed
        emb_ground = model.encode(ground,     convert_to_tensor=True)
        emb_completion = model.encode(completion, convert_to_tensor=True)

        # Cosine similarity
        score = util.cos_sim(emb_ground, emb_completion).item()

        # Print
        print(f"Entry {idx:>2}:  similarity = {score:.4f}")

        # Write CSV
        if output_csv:
            writer.writerow({'index': idx, 'similarity': f"{score:.6f}"})

    if output_csv:
        csv_file.close()
        print(f"\nResults written to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute semantic similarity between 'ground' and 'completion' using sentence embeddings."
    )
    parser.add_argument(
        'input_json',
        help="Path to input JSON file (list of {seed, ground, completion, prompt})."
    )
    parser.add_argument(
        '--model',
        default='all-MiniLM-L6-v2',
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)."
    )
    parser.add_argument(
        '--output_csv',
        default=None,
        help="If given, write results to this CSV file."
    )

    args = parser.parse_args()
    compute_similarity(args.input_json, args.model, args.output_csv)


if __name__ == "__main__":
    main()
