# Source for c-means clustering;
# https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html
# ---------------------------------------------------------------------

import skfuzzy as fuzz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


# DataSet
name1 = 'cresci'
data1 = 'Data/cresci-rtbust-2019/cresci-rtbust-2019_tweets.json'
label1 = 'Data/cresci-rtbust-2019/cresci-rtbust-2019.tsv'

name2 = 'midterm'
data2 = 'Data/midterm-2018/midterm-2018_processed_user_objects.json'
label2 = 'Data/midterm-2018/midterm-2018.tsv'

name3 = 'vendor'
data3 = 'Data/vendor-purchased-2019/vendor-purchased-2019_tweets.json'
label3 = 'Data/vendor-purchased-2019/vendor-purchased-2019.tsv'

name4 = 'verified'
data4 = 'Data/verified-2019/verified-2019_tweets.json'
label4 = 'Data/verified-2019/verified-2019.tsv'

# Experience: [features1, features2, experience_name]
followers = ['followers_count', 'followers_count', 'followers']
friends = ['friends_count', 'friends_count', 'friends']
favourites = ['favourites_count', 'favourites_count', 'favourites']
status = ['statuses_count', 'statuses_count', 'statuses']

socials = ['followers_count', 'friends_count', 'socials']
activity = ['statuses_count', 'favourites_count', 'activity']
influencers = ['followers_count', 'statuses_count', 'influencers']
influenced = ['friends_count', 'favourites_count', 'influenced']

def load_data(dataD, labelD):
    # dataframe of twitter features
    df = pd.DataFrame(columns=['id', 'followers_count', 'favourites_count', 'friends_count', 'statuses_count'])
    with open(dataD) as json_file:
        dataJson = json.loads(json_file.read())
        for p in dataJson:
            uId = int(p['user']['id'])
            followers = int(p['user']['followers_count'])
            fav = int(p['user']['favourites_count'])
            friends= int(p['user']['friends_count'])
            status = int(p['user']['statuses_count'])
            df = df.append({'id': uId, 'followers_count': followers, 'favourites_count': fav, 'friends_count': friends, 'statuses_count': status}, ignore_index=True)
    # and account labels
    df['label'] = None
    labels = {}
    with open(labelD, 'r') as labelfile:
        lines = labelfile.readlines()
        for line in lines:
            line = line.split()
            id = int(line[0])
            label = line[1]
            labels[id] = label
    df['label'] = df.apply(lambda x: labels[x['id']], axis=1)
    return df

def load_midterm(dataD, labelD):
    # dataframe of twitter features
    df = pd.DataFrame(columns=['id', 'followers_count', 'favourites_count', 'friends_count', 'statuses_count'])
    with open(dataD) as json_file:
        dataJson = json.loads(json_file.read())
        for p in dataJson:
            uId = int(p['user_id'])
            followers = int(p['followers_count'])
            fav = int(p['favourites_count'])
            friends= int(p['friends_count'])
            status = int(p['statuses_count'])
            df = df.append({'id': uId, 'followers_count': followers, 'favourites_count': fav, 'friends_count': friends, 'statuses_count': status}, ignore_index=True)
    # and account labels
    df['label'] = None
    labels = {}
    with open(labelD, 'r') as labelfile:
        lines = labelfile.readlines()
        for line in lines:
            line = line.split()
            id = int(line[0])
            label = line[1]
            labels[id] = label
    df['label'] = df.apply(lambda x: labels[x['id']], axis=1)
    return df

def load_vendor(dataD, labelD):
    # dataframe of twitter features
    df = pd.DataFrame(columns=['id', 'followers_count', 'favourites_count', 'friends_count', 'statuses_count'])
    with open(dataD) as json_file:
        dataJson = json.loads(json_file.read())
        for p in dataJson:
            uId = int(p['user']['id_str'])
            followers = int(p['user']['followers_count'])
            fav = int(p['user']['favourites_count'])
            friends= int(p['user']['friends_count'])
            status = int(p['user']['statuses_count'])
            df = df.append({'id': uId, 'followers_count': followers, 'favourites_count': fav, 'friends_count': friends, 'statuses_count': status}, ignore_index=True)
    # and account labels
    df['label'] = 'bot'
    labels = {}
    '''
    with open(labelD, 'r') as labelfile:
        lines = labelfile.readlines()
        print('readline')
        for line in lines:
            line = line.split()
            id = int(line[0])
            label = line[1]
            labels[id] = label
    # probleme with df['label'] = df.apply(lambda x: labels[x['id']], axis=1)
    # Because index 615262815 is in json but not in tsv
    df['label'] = df.apply(lambda x: labels[x['id']], axis=1)
    '''
    return df


## Configure some general styling
sns.set_style("white")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.figsize'] = (8,7)
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']



# Load data
print('load data')
df1 = load_data(data1, label1)
print('cresci done')
#df2 = load_midterm(data2, label2)
print('midterm done')
df3 = load_vendor(data3, label3)
print('vendor done')
df4 = load_data(data4, label4)
print('verified done')

##-------------------------
##-------------------------

# Set experience
nameD = 'VerifiedVendor'
df = pd.concat([df3, df4])
experience = influenced

# Create data for clustering
np.random.seed(42)  # Set seed for reproducibility
xpts = df[experience[0]].values
ypts = df[experience[1]].values

##-------------------------
##-------------------------

# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(10, 10))
alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc), size=12)
    ax.axis('off')

# Save clusters figure
fig1.tight_layout()
plt.savefig(nameD+'/'+nameD+'_clusters_'+experience[2]+'.pdf')

# Save fpc figures
fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs, color='#731810')
ax2.set_title("How Number of Clusters Change FPC?")
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")
plt.savefig(nameD +'/'+ nameD+'_fpc_'+experience[2]+'.pdf')

# Visualize bot vs human from experience perspective
def index_for_class(dataframe, c):
    df = dataframe[dataframe['label'] == c]
    index = list(df.index)
    return index

fig3, ax3 = plt.subplots()
ax3.set_title("Human vs Bot : " + experience[2])
ax3.set_xlabel(experience[0])
ax3.set_ylabel(experience[1])
for i, c in enumerate(['human', 'bot']):
    ax3.plot(xpts[index_for_class(df, c)],
            ypts[index_for_class(df, c)], '.', color=colors[i])
plt.savefig(nameD+'/' +nameD+'_BotvsHuman_'+experience[2]+'.pdf')
