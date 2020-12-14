import json

with open('Data/vendor-purchased-2019/vendor-purchased-2019_tweets.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=4, sort_keys=True))
