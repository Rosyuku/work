#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 22:13:53 2017

@author: kazuyuki
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    
    pressvec = pd.read_csv("pressvec.csv", index_col=0)
    df = pd.read_csv("data_for_w2v.csv")
    df['send_date'] = pd.to_datetime(df.send_date)
    df['year'] = df.send_date.dt.year
    df['month'] = df.send_date.dt.month
    df['day'] = df.send_date.dt.day
    df['week'] = df.send_date.dt.week
    df['weekday'] = df.send_date.dt.weekday
    pressvec = pd.concat([pressvec, df[['genre_id', 'category_id', 'year', 'month', 'day', 'week', 'weekday']]], axis=1)
    
    
    for i in range(3):
        
        p = 0.8
        np.random.seed=2
        target=df.columns[i]
        
        lnindexList = np.random.choice(df.index, size=int(df.index.shape[0]*p), replace=False)
        vdindexList = df.index[df.index.isin(lnindexList)==False].values
        
        rfr = RandomForestRegressor(n_jobs=3,
                                    n_estimators=10,
                                    )
        
        rfr.fit(pressvec.ix[df.ix[lnindexList, target].dropna().index].values, df.ix[lnindexList, target].dropna().values)
        
        result = pd.DataFrame(index=vdindexList, columns=['pred'], data=rfr.predict(pressvec.ix[vdindexList]))
        result['true'] = df.ix[vdindexList, target]
        
        result.plot.scatter(x='pred', y='true', title=target)
        print(target, 'RMSE', ((result['pred'] - result['true'])**2).mean()**0.5, ((result['pred'] - result['true'])**2).mean()**0.5 / result['true'].std())