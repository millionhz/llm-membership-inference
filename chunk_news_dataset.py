
import json


def split_news_data(obj):
    words = obj['content'].split()
    # half of the words in input and half in ground
    obj['context_length'] = len(words) // 2
    obj['input'] = ' '.join(words[:len(words) // 2])
    obj['ground'] = ' '.join(words[len(words) // 2:])
    del obj['content']
    return obj


if __name__ == '__main__':
    before = './dataset/news/newspaper_2023.json'
    after = './dataset/news/newspaper_2025.json'

    with open(before, 'r', encoding='utf-8') as f:
        before_news = json.load(f)

    with open(after, 'r', encoding='utf-8') as f:
        after_news = json.load(f)

    chunked_news = []

    for news in before_news:
        news['type'] = 'before'
        chunked_news.append(split_news_data(news))

    for news in after_news:
        news['type'] = 'after'
        chunked_news.append(split_news_data(news))

    with open('./chunked-news.json', 'w', encoding='utf-8') as f:
        json.dump(chunked_news, f)
