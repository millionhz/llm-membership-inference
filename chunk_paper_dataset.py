import json
import copy

# adjust these if you like
CONTEXT_LENGTHS = [1024, 2048, 4096]


def split_paper_data(obj, lengths=CONTEXT_LENGTHS):
    words = obj['text'].split()
    out = []
    for N in lengths:
        if len(words) < N:
            # skip if not enough words
            continue
        new_obj = copy.deepcopy(obj)
        # drop the full text
        full = words
        new_obj['context_length'] = N
        new_obj['input'] = ' '.join(full[:N])
        new_obj['ground'] = ' '.join(full[N:])
        del new_obj['text']
        del new_obj['pdf_path']
        out.append(new_obj)
    return out


if __name__ == '__main__':
    before = './dataset/papers/before/papers-extracted.json'
    after = './dataset/papers/after/papers-extracted.json'

    with open(before, 'r', encoding='utf-8') as f:
        before_papers = json.load(f)

    with open(after, 'r', encoding='utf-8') as f:
        after_papers = json.load(f)

    before_papers = before_papers[:50]
    after_papers = after_papers[:50]

    chunked_papers = []
    for paper in before_papers:
        paper['type'] = 'before'
        chunked_papers.extend(split_paper_data(paper))

    for paper in after_papers:
        paper['type'] = 'after'
        chunked_papers.extend(split_paper_data(paper))

    with open('./chunked-papers.json', 'w', encoding='utf-8') as f:
        json.dump(chunked_papers, f)
