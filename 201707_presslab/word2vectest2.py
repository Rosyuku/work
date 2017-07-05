#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 21:24:46 2017

@author: kazuyuki
"""

from gensim.models.word2vec import Word2Vec

if __name__ == "__main__":
    model_path = "latest-ja-word2vec-gensim-model/word2vec.gensim.model"
    model = Word2Vec.load(model_path)