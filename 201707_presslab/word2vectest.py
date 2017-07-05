#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 22:27:24 2017


@author: kazuyuki
"""

import MeCab

if __name__ == "__main__":
    
    mecab = MeCab.Tagger("-Ochasen")
    mecab = MeCab.Tagger("mecabrc")
    print(mecab.parse("MecabをインストールしてPythonから使ってみた"))
