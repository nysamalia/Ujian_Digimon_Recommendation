import numpy as np
import pandas as pd

df = pd.read_json('digimon.json')
# print(df.head())
# print(df.isnull().sum())
# print(df.columns)

df = df[['stage', 'type', 'attribute', 'digimon', 'image']]
def kombinasi (i):
    return str(i['stage']) + '$' + str(i['type']) + '$' + str(i['attribute'])

df['x'] = df.apply(kombinasi, axis=1)
# print(df.head()) 

from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    tokenizer=lambda i: i.split('$')
)

kategori = model.fit_transform(df['x'])
# print(model.get_feature_names())


from sklearn.metrics.pairwise import cosine_similarity
favDigi = 'Agumon'
ifavDigi = df[df['digimon'] == favDigi].index.values[0]
cosScore = cosine_similarity(kategori)
# print(cosScore)

fav = {
        'digimon': favDigi,
        'stage': df[df['digimon']== favDigi]['stage'].values[0],
        'type':df[df['digimon']== favDigi]['type'].values[0],
        'attribute':df[df['digimon']== favDigi]['attribute'].values[0],
        'image':df[df['digimon']== favDigi]['image'].values[0],

}
# print(fav['stage'])

listDigi = list(enumerate(cosScore[ifavDigi]))
sortDigi = sorted(listDigi, key=lambda x: x[1], reverse=True)
# print(sortDigi)

rek=[]
for i in sortDigi[:7]:
    if df.iloc[i[0]]['digimon'] != favDigi:
        x = {
            'digimon' : df.iloc[i[0]]['digimon'],
            'stage' : df.iloc[i[0]]['stage'],
            'tipe' : df.iloc[i[0]]['type'],
            'attribute' : df.iloc[i[0]]['attribute'],
            'image' : df.iloc[i[0]]['image']
        }
        rek.append(x)

rek=[]
for i in sortDigi[:7]:
    if df.iloc[i[0]]['digimon'] != favDigi:
        x = {
            'digimon' : df.iloc[i[0]]['digimon'],
            'stage' : df.iloc[i[0]]['stage'],
            'tipe' : df.iloc[i[0]]['type'],
            'attribute' : df.iloc[i[0]]['attribute'],
            'image' : df.iloc[i[0]]['image']
        }
        rek.append(x)
print(rek)


