import json
import numpy as np
from multiprocessing import Pool
from sklearn.cluster import KMeans
from functools import partial
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy import spatial

# Constants/Tweakable values
CLUSTERS = 150
NUM_THREADS = -1
SENTENCE_WORD_WEIGHT = 10

def get_unique_words(data):
    unique_words = set()
    for subreddit, word_count in data:
        for word, count in word_count:
            if word not in unique_words:
                unique_words.add(word)
    return list(unique_words)  # Convert set to numpy array


def get_histogram_from_sentence(unique_words, word_list):
    histogram = np.zeros(len(unique_words))
    for word in word_list:
        if word in unique_words:
            histogram[unique_words.index(word)] += SENTENCE_WORD_WEIGHT
    histogram /= 100
    return histogram


def __get_histogram_words(unique_words, subreddit_word_count):
    histogram = np.zeros(len(unique_words))
    name, word_count = subreddit_word_count
    # print('Processing', name)
    for word, count in word_count:
        if word in unique_words:
            histogram[unique_words.index(word)] = count
    histogram /= 100
    return [name, histogram]


def __get_histogram_groups(centroids, subreddit):
    name, histogram = subreddit
    weights = []
    for centroid in centroids:
        # Get rid of warning
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = 1 - spatial.distance.cosine(centroid, histogram)
        weights.append(weight)
    weights = np.asarray(weights)
    # Sort indexes of weights using the values stored at the index
    weighted_groups = np.argsort(weights)[:-1]
    return [name, weighted_groups.tolist()]


def cluster(unique_words, data, save_to_file):
    # Generate histogram for each subreddit
    print('Generating histograms:')
    pool = Pool()
    func = partial(__get_histogram_words, unique_words)
    subreddit_histograms = []
    for histogram in tqdm(pool.imap_unordered(func, data), total=len(data)):
        subreddit_histograms.append(histogram)

    # clump all the histograms into one array
    print('Clumping histograms')
    big_ass_array = np.empty((len(data), len(unique_words)))
    itterator = 0
    for subreddit, histogram in subreddit_histograms:
        big_ass_array[itterator] = histogram
        itterator += 1

    # KMeans
    print('Running KMeans (This takes a while)')
    kmeans = KMeans(n_clusters=CLUSTERS, n_jobs=NUM_THREADS).fit(big_ass_array)

    # How'd we do?
    print('Grouping subreddits')
    groups = []
    func = partial(__get_histogram_groups, kmeans.cluster_centers_)
    for histogram_group in tqdm(pool.imap_unordered(func, subreddit_histograms),
                                total=len(subreddit_histograms)):
        groups.append(histogram_group)

    pool.close()
    pool.join()

    # Save to json file
    if save_to_file:
        # Sort groups for debugging purposes
        groups.sort(key=lambda tup: tup[1][0], reverse=True)
        print('Writing to files')
        with open('uniqueWords.json', 'w') as out:
            json.dump(unique_words, out)
        with open('groups.json', 'w') as out:
            json.dump(groups, out)
        with open('centroids.json', 'w') as out:
            json.dump(kmeans.cluster_centers_.tolist(), out)
        out.close()

    return groups

if __name__ == '__main__':
    with open('relevantTerms.json') as data_file:
        data = json.load(data_file)
    data_file.close()
    # Get unique words from data
    print('Finding unique words')
    unique_words = get_unique_words(data)
    cluster(unique_words, data, save_to_file=True)
