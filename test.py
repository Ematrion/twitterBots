import valuelimit
import json
import pandas as pd
import numpy as np
import skfuzzy as fuzz

import processtwitter as pt


# DataSet
name1 = 'cresci'
data1 = 'Data/cresci-rtbust-2019/cresci-rtbust-2019_tweets.json'
label1 = 'Data/cresci-rtbust-2019/cresci-rtbust-2019.tsv'

# Experience: [features1, features2, experience_name]
followers = ['followers_count', 'followers_count', 'followers', 6]
friends = ['friends_count', 'friends_count', 'friends', 5]
favourites = ['favourites_count', 'favourites_count', 'favourites', 5]
status = ['statuses_count', 'statuses_count', 'statuses', 6]

socials = ['followers_count', 'friends_count', 'socials', 8]
activity = ['statuses_count', 'favourites_count', 'activity', 5]
influencers = ['followers_count', 'statuses_count', 'influencers', 6]
influenced = ['friends_count', 'favourites_count', 'influenced', 5] # bad features regarding bot vs human

experiences = [followers, friends, favourites, status, socials, activity, influencers, influenced]
features = [x[2] for x in experiences]


# load Raw Twitter data
df = pt.load_data(data1, label1)

# Set seed for reproducibility
np.random.seed(42)

# process data
dweka = pd.DataFrame(index = df.index, columns = features)
for exp in experiences:
    # Create data for clustering
    xpts = df[exp[0]].values
    ypts = df[exp[1]].values
    alldata = np.vstack((xpts, ypts))

    ncenters = exp[3]
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    clusters = np.argmax(u, axis=0)
    dweka[exp[2]] =clusters

dweka['class'] = df['label']
pt.makeWekaFile(name1, dweka)
