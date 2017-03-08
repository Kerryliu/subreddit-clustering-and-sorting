from glob import glob
import json
import parser


def get_top_subreddits(parsed_sentence, relevant_terms):
    ranks = []
    for subreddit_word_count in relevant_terms:
        weight = 0
        name, word_count = subreddit_word_count
        word_count = dict(word_count)
        for word, count in parsed_sentence.items():
            if word in word_count:
                weight += count * word_count[word]
        if weight != 0:
            ranks.append((name, weight))
    ranks.sort(key=lambda tup: tup[1], reverse=True)
    return ranks


def classify(relevant_terms, sentence):
    word_count = parser.sentence_to_word_dict(sentence)
    result = get_top_subreddits(word_count, relevant_terms)
    return result

if __name__ == '__main__':
    sentence = 'Teach me how to program Haskell'
    with open('relevantTerms.json') as data_file:
        relevant_terms = json.load(data_file)
    data_file.close()
    classify(relevant_terms, sentence)
