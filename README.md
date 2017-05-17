# Sub reddit clustering and sorting

 - parser.py parses the data from each subreddit, generating a list of relevant words that corresponds to that subreddit.  
 - Once training is complete, classify.py can be used to classify sentences to subreddits based on similarity.  
 - magic.py was an attempt to cluster subreddits to reduce the number of categories, but failed to be of use, as most subreddits share little similarity between each other.  

Data is from:
https://github.com/umbrae/reddit-top-2.5-million
Install requirements and run main.py
Requires around ~8 gigs of ram...
