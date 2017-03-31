from glob import glob
import json
import parser
import pickle


def get_top_subreddits_dict(parsed_sentence, relevant_terms_dict):
    ranks_dict = {}
    for word, count in parsed_sentence.items():
        if word in relevant_terms_dict:
            for subreddit, count in relevant_terms_dict[word]:
                if subreddit not in ranks_dict:
                    ranks_dict[subreddit] = count
                else:
                    ranks_dict[subreddit] += count
    ranks = list(ranks_dict.items())
    ranks.sort(key=lambda tup: tup[1], reverse=True)
    return ranks



def create_dict(relevant_terms, save_to_file=False):
    relevant_terms_dict = {}
    for subreddit_word_count in relevant_terms:
        name, word_count = subreddit_word_count
        for word, count, in word_count:
            if word not in relevant_terms_dict:
                relevant_terms_dict[word] = [(name, count)]
            else:
                relevant_terms_dict[word].append((name, count))
    print(relevant_terms_dict)
    if save_to_file:
        pickle.dump(relevant_terms_dict, open('relevant_terms_dict.p', 'wb'))
    return relevant_terms_dict


def classify(relevant_terms, sentence):
    word_count = parser.sentence_to_word_dict(sentence)
    result = get_top_subreddits_dict(word_count, relevant_terms)
    return result

if __name__ == '__main__':
    sentence = 'Teach me how to program Haskell'
    with open('relevant_terms_dict.p', 'rb') as data_file:
        relevant_terms_dict = pickle.load(data_file)
    data_file.close()
    print(classify(relevant_terms_dict, sentence))
