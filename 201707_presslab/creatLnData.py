#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 23:09:40 2017

@author: kazuyuki
"""

import pandas as pd
import MeCab
from io import StringIO
import re
from joblib import Parallel, delayed
from gensim.models import word2vec
from sklearn import preprocessing as sp

def vectorize(docs, model):
    
    try:
        
        mecab = MeCab.Tagger("-Ochasen")
        data = mecab.parse(docs)
        data = StringIO(data.replace("\"", ""))
        data = pd.read_csv(data, sep='\t', header=None)
        data = data.loc[(data[3].str.find("名詞") >= 0) & (data[3].str.find("接頭詞") < 0) & (data[3].str.find("サ変接続") < 0)]
        data = data[data[0].isin(pd.DataFrame.from_dict(model.wv.vocab, orient='index').index)]
        data = data[[0]]
        
        vec = pd.DataFrame(data.applymap(model.wv.word_vec)[0].values.mean(axis=0))
        
        return vec
    
    except:
        
        return []


if __name__ == "__main__":
    
    df = pd.read_csv("data_for_w2v.csv")
    
    model = word2vec.Word2Vec.load("word2vec.gensim.model")
    
    df['text'] = df['title'] + "\n" + df['summary'] + "\n" + df['content']
    
    pressvec = pd.concat(Parallel(n_jobs=3, verbose=10)([delayed(vectorize)(docs, model) for docs in df['text'].str.lower().values]), axis=1).T
    pressvec.index = df.index

    pressvec.to_csv("pressvec.csv")
