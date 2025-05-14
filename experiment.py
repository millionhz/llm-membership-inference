from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

BASE_PROMPT = 'You are an assistant that uses source material exactly as it was written with no paraphrasing, no summarising, and no commentary. When provided with a text, you will complete the text based on your memory of the text. You will not add any new information or change the meaning of the text. You will only complete the text with the most likely next word or phrase based on your training data. Your response should *only* be the completion of the text.'

MODEL = "gpt-4o-2024-08-06"
# MODEL = "gpt-4o-mini-2024-07-18"

OUT_DIR = Path("completions")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def context_length_experiment(dataset, temperature):
    T = temperature

    with open(dataset, "r", encoding="utf-8") as f:
        records = json.load(f)

    with open(f'./chunked-papers-completions-temperature-{T}.jsonl', 'a') as f:
        for rec in tqdm(records, desc=f"Processing papers", unit="record"):
            response = client.responses.create(
                model=MODEL,
                instructions=BASE_PROMPT,
                input=rec["input"],
                temperature=T,)

            rec['completion'] = response.output_text
            rec['temperature'] = T
            rec['prompt'] = BASE_PROMPT

            json.dump(rec, f)
            f.write("\n")

            time.sleep(1)  # Avoid rate limit


def temperature_experiment(dataset, temperature):
    T = temperature

    with open(dataset, "r", encoding="utf-8") as f:
        records = json.load(f)

    with open(f'./chunked-news-completions-temperature-{T}.jsonl', 'a') as f:
        for rec in tqdm(records, desc=f"Processing temperature {T}", unit="record"):
            response = client.responses.create(
                model=MODEL,
                instructions=BASE_PROMPT,
                input=rec["input"],
                temperature=T,)

            rec['completion'] = response.output_text
            rec['temperature'] = T
            rec['prompt'] = BASE_PROMPT

            json.dump(rec, f)
            f.write("\n")


def masking_experiment(dataset, temperature=0.1):
    MASK_COMPLETION_PROMPT = 'You are a factual information retrieval assistant. You are given input text with one or more missing pieces of information, represented by the token <MASK>. Your task is to replace each <MASK> with the most accurate and specific word based on your internal memory. In your response, provide the passage with the <MASK> tokens replaced by the most likely next word. Make your best effort to fill in the masked words. Make sure to not use a code block or any other formatting. Your response should only contain the completion of the text.'

    T = temperature

    with open(dataset, "r", encoding="utf-8") as f:
        records = json.load(f)

    with open(f'./masked-news-completions-temperature-{T}.jsonl', 'a') as f:
        for rec in tqdm(records, desc=f"Processing Masked Dataset", unit="record"):
            response = client.responses.create(
                model=MODEL,
                instructions=MASK_COMPLETION_PROMPT,
                input=rec["input"],
                temperature=T,)

            rec['completion'] = response.output_text
            rec['temperature'] = T
            rec['prompt'] = MASK_COMPLETION_PROMPT

            json.dump(rec, f)
            f.write("\n")


if __name__ == "__main__":
    # NEWS_DATASET = Path("dataset/news/chunked-news.json")
    # PAPER_DATASET = Path("dataset/papers/chunked-papers-50.json")

    # MASKED_PAPER_DATASET_25 = Path("dataset/papers/masked-papers-0.25.json")
    # MASKED_PAPER_DATASET_50 = Path("dataset/papers/masked-papers-0.5.json")
    # MASKED_PAPER_DATASET_30 = Path("dataset/papers/masked-papers-0.3.json")

    # MASKED_ABSTRACT_PAPER_DATASET_30 = Path(
    #     "dataset/papers/masked-abstract-papers-0.3.json")
    # MASKED_ABSTRACT_PAPER_DATASET_50 = Path(
    #     "dataset/papers/masked-abstract-papers-0.5.json")

    MASKED_NEWS_DATASET_25 = Path('./dataset/news/masked-news-0.25.json')
    MASKED_NEWS_DATASET_30 = Path('./dataset/news/masked-news-0.3.json')
    MASKED_NEWS_DATASET_40 = Path('./dataset/news/masked-news-0.4.json')
    MASKED_NEWS_DATASET_50 = Path('./dataset/news/masked-news-0.5.json')

    # temperature_experiment(NEWS_DATASET, 0.5)
    # context_length_experiment(PAPER_,DATASET, 0.5)
    # masking_experiment(MASKED_NEWS_DATASET_25, temperature=0.1)
    # masking_experiment(MASKED_NEWS_DATASET_30, temperature=0.1)
    # masking_experiment(MASKED_NEWS_DATASET_40, temperature=0.1)
    # masking_experiment(MASKED_NEWS_DATASET_50, temperature=0.1)
    masking_experiment(MASKED_NEWS_DATASET_30, temperature=0.1)
