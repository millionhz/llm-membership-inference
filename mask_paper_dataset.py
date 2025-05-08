import json
import copy
import random

MASK_PROB = 0.3
QUIZ_LENGTH = 1024


def mask_paper(paper):
    words = paper['text'].split()

    # start from the middle of the text
    start = len(words) // 2
    # only use QUIZ_LENGTH words
    ground = words[start:start + QUIZ_LENGTH]
    input = ground.copy()

    # for each word, add <mask> with probability MASK_PROB
    for i in range(len(input)):
        if random.random() < MASK_PROB:
            input[i] = '<MASK>'

    paper['mask_prob'] = MASK_PROB
    paper['context_length'] = QUIZ_LENGTH
    paper['input'] = ' '.join(input)
    paper['ground'] = ' '.join(ground)
    del paper['text']
    del paper['pdf_path']
    return paper


if __name__ == '__main__':
    before = './dataset/papers/before/papers-extracted.json'
    after = './dataset/papers/after/papers-extracted.json'

    with open(before, 'r', encoding='utf-8') as f:
        before_papers = json.load(f)

    with open(after, 'r', encoding='utf-8') as f:
        after_papers = json.load(f)

    before_papers = before_papers[:50]
    after_papers = after_papers[:50]

    masked_papers = []
    for paper in before_papers:
        paper['type'] = 'before'
        masked_papers.append(mask_paper(paper))

    for paper in after_papers:
        paper['type'] = 'after'
        masked_papers.append(mask_paper(paper))

    with open('./masked-papers.json', 'w', encoding='utf-8') as f:
        json.dump(masked_papers, f)
