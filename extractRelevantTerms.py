import glob
import os
import csv
import operator
import json
from multiprocessing import Pool
from functools import partial
import spacy

# Spacy~
nlp = spacy.load('en')

# Constants/Tweakable values
SUBREDDIT_MIN_WORD_COUNT = 20  # How common a word needs to be for it to count
SHARED_WORD_COUNT = 500  # How isolated a word is to a subreddit for it to count


def load_from_CSV(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        raw_data = list(reader)

    raw_data.pop(0)  # Remove header
    filename = os.path.basename(path)
    name, extension = os.path.splitext(filename)  # Get the subreddit

    # Append each post to posts
    raw_sentences = ''
    for raw_post_data in raw_data:
        title = raw_post_data[4]
        self_text = raw_post_data[9]
        raw_sentences += title + ' ' + self_text
    return [name, raw_sentences]


def get_word_count(subreddit):
    name, raw_sentences = subreddit
    print('Processing:', name)
    word_count = dict()
    doc = nlp(raw_sentences)
    for token in doc:
        # Grab almost everything for now
        if not (token.is_punct or token.like_email
                or token.like_url or token.is_space or len(token.text) > 50):
            word = token.lower_
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
    # Remove oddballs by pushing good words to list
    better_word_count = []  # Best variable name ever
    for key, value in word_count.items():
        if value >= SUBREDDIT_MIN_WORD_COUNT:
            better_word_count.append([key, value])
    # Sort to make my life easier when debugging
    better_word_count.sort(key=lambda tup: tup[1], reverse=True)
    return [name, better_word_count]


def remove_common_words(common_words, subreddit_word_count):
    good_words = []
    name, word_count = subreddit_word_count
    for word, count in word_count:
        if word not in common_words:
            good_words.append([word, count])
    return [name, good_words]


def main():
    paths = glob.glob('./data/*.csv')
    pool = Pool()
    # Move csv data into posts list
    subreddits = pool.map(load_from_CSV, paths)

    # Get word count of each subreddit
    subreddit_word_counts = pool.map(get_word_count, subreddits)

    # Global common word count.  Common words between subreddits
    print('Finding common words')
    common_words = []
    shared_word_count = dict()
    for name, word_count in subreddit_word_counts:
        for word, count in word_count:
            if word not in shared_word_count:
                shared_word_count[word] = 1
            else:
                shared_word_count[word] += 1
    for key, value in shared_word_count.items():
        if value >= SHARED_WORD_COUNT:
            common_words.append(key)  # No need for count
    # Print out common words for debugging
    # common_words.sort(key=lambda tup: tup[1], reverse=True)

    func = partial(remove_common_words, common_words)
    final_subreddit_word_counts = pool.map(func, subreddit_word_counts)

    print('Writing to file')
    with open('relevantTerms.json', 'w') as outfile:
        json.dump(final_subreddit_word_counts, outfile)

if __name__ == '__main__':
    main()
