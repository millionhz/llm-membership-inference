import json
import random

MASK_PROB = 0.4
QUIZ_LENGTH = 1024


def mask_abstract(paper):
    ground = paper['abstract'].split()
    input = ground.copy()

    # for each word, add <mask> with probability MASK_PROB
    for i in range(len(input)):
        if random.random() < MASK_PROB:
            input[i] = '<MASK>'

    paper['mask_prob'] = MASK_PROB
    paper['context_length'] = len(input)
    paper['input'] = ' '.join(input)
    paper['ground'] = ' '.join(ground)
    del paper['abstract']
    del paper['text']
    del paper['pdf_path']
    return paper


def mask_text(paper):
    words = paper['content'].split()

    ground = words
    input = ground.copy()

    # for each word, add <mask> with probability MASK_PROB
    for i in range(len(input)):
        if random.random() < MASK_PROB:
            input[i] = '<MASK>'

    paper['mask_prob'] = MASK_PROB
    paper['context_length'] = QUIZ_LENGTH
    paper['input'] = ' '.join(input)
    paper['ground'] = ' '.join(ground)
    del paper['content']
    return paper


def text():
    before = './dataset/news/newspaper_2023.json'
    after = './dataset/news/newspaper_2025.json'

    with open(before, 'r', encoding='utf-8') as f:
        before = json.load(f)

    with open(after, 'r', encoding='utf-8') as f:
        after = json.load(f)

    masked = []
    for paper in before:
        paper['type'] = 'before'
        masked.append(mask_text(paper))

    for paper in after:
        paper['type'] = 'after'
        masked.append(mask_text(paper))

    with open(f'./masked-news-{MASK_PROB}.json', 'w', encoding='utf-8') as f:
        json.dump(masked, f)


if __name__ == '__main__':
    text()
