from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import json
import os
from dotenv import load_dotenv

load_dotenv()


MODEL = "gpt-4o-2024-08-06"
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8]
OUT_DIR = Path("completions")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def temperature_experiment(dataset, temperature):
    TEMPERATURE_PROMPT = 'You are an assistant that uses source material exactly as it was written with no paraphrasing, no summarising, and no commentary. When provided with a text, you will complete the text based on your memory of the text. You will not add any new information or change the meaning of the text. You will only complete the text with the most likely next word or phrase based on your training data. Your response should *only* be the completion of the text.'

    T = temperature

    with open(dataset, "r", encoding="utf-8") as f:
        records = json.load(f)

    with open(f'./chunked-news-completions-temperature-{T}.jsonl', 'a') as f:
        for rec in tqdm(records, desc=f"Processing temperature {T}", unit="record"):
            response = client.responses.create(
                model=MODEL,
                instructions=TEMPERATURE_PROMPT,
                input=rec["input"],
                temperature=T,)

            rec['completion'] = response.output_text
            rec['temperature'] = T
            rec['prompt'] = TEMPERATURE_PROMPT

            json.dump(rec, f)
            f.write("\n")


if __name__ == "__main__":
    NEWS_DATASET = Path("dataset/news/chunked-news.json", 0.5)
    PAPER_DATASET = Path("dataset/papers/chunked-papers.json")

    temperature_experiment(NEWS_DATASET)
