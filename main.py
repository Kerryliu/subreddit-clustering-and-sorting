import parser
import magic
import os.path
from glob import glob
import json
paths = glob('./data/*.csv')


def main():
    if (os.path.exists('uniqueWords.json') and
            os.path.exists('centroids.json') and os.path.exists('groups.json')):
        print('---~Extracting relevant terms~---')
        data = parser.get_subreddit_word_counts(paths, save_to_file=True)
        print('---~Clustering~---')
        unique_words, groups, centroids = magic.cluster(
            data, save_to_file=True)
    else:
        with open('uniqueWords.json') as data_file:
            unique_words = json.load(data_file)
        with open('centroids.json') as data_file:
            centroids = json.load(data_file)
        with open('groups.json') as data_file:
            groups = json.load(data_file)
        data_file.close()

    sentence = 'linux linux linux'
    parsed_sentence = parser.sentence_to_word_dic(sentence)
    sentence_histogram = magic.get_histogram_from_sentence(
        unique_words, parsed_sentence)
    asdf = ['test', sentence_histogram]
    histogroups = magic.get_histogram_groups(centroids, asdf)
    print(histogroups)


if __name__ == '__main__':
    main()
