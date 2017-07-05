#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 22:13:02 2017

@author: kazuyuki
"""

import pandas as pd
import MeCab
from io import StringIO
import re
from joblib import Parallel, delayed
from gensim.models import word2vec

def tokenize(sentence):
    
    try:
        
        mecab = MeCab.Tagger("-Ochasen")
        data = mecab.parse(sentence)
        data = StringIO(data)
        data = pd.read_csv(data, sep='\t', header=None)
        data = data.loc[(data[3].str.find("名詞") >= 0) & (data[3].str.find("接頭詞") < 0) & (data[3].str.find("サ変接続") < 0)]
        texts = list(data[0])
        
        return texts
    
    except:
        
        return []

if __name__ == "__main__":
    
    df = pd.read_csv("data_for_w2v.csv")
    
    textdf = pd.melt(df, value_vars=['title', 'summary', 'content'])
    docs = re.split("[\n]", ('\n'.join(textdf['value'].str.lower().values)).replace("\u3000",""))
    
    tokeData = Parallel(n_jobs=3, verbose=10)([delayed(tokenize)(s) for s in docs])
    
    model = word2vec.Word2Vec(tokeData,
                              size=100,
                              min_count=5,
                              window=5,
                              iter=5)
    
    model.save("word2vec.gensim.model")
    