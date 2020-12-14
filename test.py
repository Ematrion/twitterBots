import valuelimit
import json
import pandas as pd
import numpy as np



# dataframe of twitter features
df = pd.DataFrame(columns=['id', 'followers_count', 'favourites_count', 'friends_count', 'statuses_count'])
with open('Data/verified-2019/verified-2019_tweets.json') as json_file:
    dataJson = json.loads(json_file.read())
    for p in dataJson:
        #uId = int(p['user']['id'])
        followers = int(p['user']['followers_count'])
        fav = int(p['user']['favourites_count'])
        friends= int(p['user']['friends_count'])
        status = int(p['user']['statuses_count'])
        #df = df.append({'id': uId, 'followers_count': followers, 'favourites_count': fav, 'friends_count': friends, 'statuses_count': status}, ignore_index=True)
        df = df.append({'followers_count': followers, 'favourites_count': fav, 'friends_count': friends, 'statuses_count': status}, ignore_index=True)



# Compute class border/thresholds for every used features
for col in df.columns:
    print(col)
    #print(valuelimit.linear(df[col], 5))
    #print(valuelimit.uniform(df[col], 5))
    print(valuelimit.cMeans(df[col], 5))
