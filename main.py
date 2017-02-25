import parser
import magic
import os.path
from glob import glob
import json
paths = glob('./data/*.csv')

if os.path.exists('groups.json'):
    print('---~Extracting relevant terms~---')
    data = parser.get_subreddit_word_counts(paths, save_to_file=True)
    print('---~Clustering~---')
    unique_words = magic.get_unique_words(data)
    magic.cluster(unique_words, data, save_to_file=True)

sentence = 'I would like a potato'
