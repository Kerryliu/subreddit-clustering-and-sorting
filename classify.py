from glob import glob
import json
import parser
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def get_top_subreddits(parsed_sentence, relevantTerms):
    pool = Pool()
    ranks = []
    func = partial(__compare_to_subreddit, parsed_sentence)
    for subreddit_weight in tqdm(pool.imap_unordered(
            func, relevantTerms), total=len(relevantTerms)):
        if subreddit_weight[1] != 0:
            ranks.append(subreddit_weight)
    pool.close()
    pool.join()
    ranks.sort(key=lambda tup: tup[1], reverse=True)
    return ranks


def __compare_to_subreddit(parsed_sentence, subreddit_word_count):
    weight = 0
    name, word_count = subreddit_word_count
    word_count = dict(word_count)
    for word, count in parsed_sentence.items():
        if word in word_count:
            weight += count * word_count[word]
    return (name, weight)


def classify(relevant_terms, sentence):
    word_count = parser.sentence_to_word_dic(sentence)
    result = get_top_subreddits(word_count, relevant_terms)
    return result

if __name__ == '__main__':
    sentence = 'Teach me how to program Haskell'
    with open('relevantTerms.json') as data_file:
        relevant_terms = json.load(data_file)
    data_file.close()
    classify(relevant_terms, sentence)
