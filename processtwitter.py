import pandas as pd
import json

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
    return df


def arffRow(row):
    arffLine = ''
    for feature in row.values:
        arffLine += str(feature) + ','
    arffLine = arffLine[:-1]
    arffLine += '\n'
    return arffLine

def makeWekaFile(title, dataframe):
    with open(title +'.arff', 'w+') as arff:
        # Hearder
        arff.write('% ' + title + '%')
        arff.write('\n')
        arff.write('@RELATION twitterBots')
        arff.write('\n')
        # Attributes
        for col in dataframe.columns:
            arffAttr = '@ATTRIBUTE '
            arffAttr += str(col) + ' {'
            values = pd.unique(dataframe[col]).tolist()
            values = str(values)
            values = values[:-1]
            values = values[1:]
            arffAttr += values
            arffAttr += '}'
            arff.write(arffAttr)
            arff.write('\n')
        # Insert data line
        arff.write('@DATA')
        arff.write('\n')
        dataframe.apply(lambda x: arff.write(arffRow(x)), axis=1)
