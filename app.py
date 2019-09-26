import json, requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, redirect, request, abort
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
@app.route('/home')


def home():
        return render_template('home.html')

@app.route('/hasil', methods=['POST','GET'])
def hasil():
    if request.method == 'POST':
        df = pd.read_json('digimon.json')
        diginame = request.form['diginame'].capitalize()
        if str(diginame) == '<Response [404]>':
            return render_template('error.html')
        else:
            df = df[['stage', 'type', 'attribute', 'digimon', 'image']]
            def kombinasi (i):
                return str(i['stage']) + '$' + str(i['type']) + '$' + str(i['attribute'])

            df['x'] = df.apply(kombinasi, axis=1)

            model = CountVectorizer(tokenizer=lambda i: i.split('$'))
            kategori = model.fit_transform(df['x'])

            favDigi = diginame
            ifavDigi = df[df['digimon'] == favDigi].index.values[0]
            cosScore = cosine_similarity(kategori)

            listDigi = list(enumerate(cosScore[ifavDigi]))
            sortDigi = sorted(listDigi, key=lambda x: x[1], reverse=True)
            
            fav = {
                'digimon': favDigi,
                'stage': df[df['digimon']== favDigi]['stage'].values[0],
                'tipe':df[df['digimon']== favDigi]['type'].values[0],
                'attribute':df[df['digimon']== favDigi]['attribute'].values[0],
                'image':df[df['digimon']== favDigi]['image'].values[0]
            }
            
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
                    # print(rek)
                    return render_template ('hasil.html', rek=rek, fav=fav)  
    else:
        abort(404)

@app.errorhandler(404)
def page_not_found(error):
	return render_template('error.html')

if __name__ == "__main__":
    app.run(debug=True)