from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

BASE_PROMPT = 'You are an assistant that uses source material exactly as it was written with no paraphrasing, no summarising, and no commentary. When provided with a text, you will complete the text based on your memory of the text. You will not add any new information or change the meaning of the text. You will only complete the text with the most likely next word or phrase based on your training data. Your response should *only* be the completion of the text.'

BASE_PROMPT2 = "Hmm, let's play an interesting game. I am going to give you part of a text, and you will have to complete it. You can only use the text I give you to complete it. You can't add any new information or change the meaning of the text. You can only complete the text with the most likely next word or phrase. I will then be evaluating your response based on the true data I have Your response should *only* be the completion of the text."

MODEL = "gpt-4o-2024-08-06"
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


if __name__ == "__main__":
    NEWS_DATASET = Path("dataset/news/chunked-news.json")
    PAPER_DATASET = Path("dataset/papers/chunked-papers.json")

    # temperature_experiment(NEWS_DATASET, 0.5)
    context_length_experiment(PAPER_DATASET, 0.5)
