import json
from multiprocessing import Pool
from sklearn.cluster import KMeans
import numpy as np
from functools import partial
from sklearn.decomposition import PCA
from scipy import spatial

# Constants/Tweakable values
CLUSTERS = 100
NUM_TOP_GROUPS = 5


def get_histogram(unique_words, subreddit_word_count):
    histogram = np.zeros(len(unique_words))
    name, word_count = subreddit_word_count
    print('Processing', name)
    for word, count in word_count:
        if word in unique_words:
            histogram[unique_words.index(word)] = count
    histogram /= 100
    return [name, histogram]


def main():
    pool = Pool()

    with open('relevantTerms.json') as data_file:
        data = json.load(data_file)

    # Get unique words from data
    unique_words = set()
    for subreddit, word_count in data:
        for word, count in word_count:
            if word not in unique_words:
                unique_words.add(word)
    unique_words = list(unique_words)  # Convert set to numpy array

    # Generate histogram for each subreddit
    func = partial(get_histogram, unique_words)
    histograms = pool.map(func, data)

    # clump all the histograms into one array
    big_ass_array = np.empty((len(data), len(unique_words)))
    itterator = 0
    for subreddit, histogram in histograms:
        big_ass_array[itterator] = histogram
        itterator += 1

    # KMeans
    print('Running KMeans')
    kmeans = KMeans(n_clusters=CLUSTERS, n_jobs=-1).fit(big_ass_array)
    print(kmeans.cluster_centers_.shape)

    # How'd we do?
    print('Grouping subreddits')
    groups = []
    for name, histogram in histograms:
        weights = []
        for centroid in kmeans.cluster_centers_:
            # Get rid of warning
            with np.errstate(divide='ignore', invalid='ignore'):
                weight = 1 - spatial.distance.cosine(centroid, histogram)
            weights.append(weight)
        weights = np.asarray(weights)
        weighted_groups = np.argsort(weights)[:-1]
        top_groups = weighted_groups[:NUM_TOP_GROUPS].tolist()
        groups.append([name, top_groups])
    groups.sort(key=lambda tup: tup[1][0], reverse=True)

    # Save to json file
    print('Writing to file')
    with open('groups.json', 'w') as outfile:
        json.dump(groups, outfile)

if __name__ == '__main__':
    main()
