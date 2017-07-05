#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 21:39:18 2017

@author: kazuyuki
"""

import pandas as pd
import MeCab
from io import StringIO
import re
from joblib import Parallel, delayed
from gensim.models import word2vec
import itertools
import sys

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
    
    docs = '\n'.join(df['content'].values)
#    example = df.loc[0, 'content'].lower()

    for i, docs in enumerate (df['content'].values):
#        print(i)
        i = 3017

        try:
            example = re.split("[\n]", docs.replace("\u3000",""))
            
            tokeData = Parallel(n_jobs=1, verbose=0)([delayed(tokenize)(s) for s in example])
            
            for t in tokeData:
                if len(t) < 1:
                    del tokeData[tokeData.index(t)]
            
            model = word2vec.Word2Vec(tokeData, min_count=1, )
        except:
            print(i, docs[:20])
            print(sys.exc_info())
            
        break
    
#    tokenize(example[0])
    
#    mecab = MeCab.Tagger("-Ochasen")
#    
#    print(mecab.parse(example[0]))
#    data = mecab.parse(example[0])
#    
#    data = StringIO(data)
#    
#    data = pd.read_csv(data, sep='\t', header=None)
#    
#    data = data.loc[(data[3].str.find("名詞") >= 0) & (data[3].str.find("接頭詞") < 0)]
