#!/usr/bin/env python
# -*- coding: utf-8 -*-

# entropy_functions.py
# (c) Zexun Chen, 2020-04-09
# sxtpy2010@gmail.com

# import pandas as pd
import modin.pandas as pd


def geo2id(df, resolution=None, lat='lat', lon='lon'):
    """ convert lat and lon to a geo-id
    :param df: dataframe or series with geo-coordinate information
    :param lon: longitude
    :param lat: latitude
    :param resolution: what the resolution used when taking geo-id
    :return: dataframe with geo-id
    """
    if resolution is not None:
        df = pd.DataFrame(df)
        df['lat'] = df['lat'].map(lambda x: round(x, resolution))
        df['lon'] = df['lon'].map(lambda x: round(x, resolution))
    df_new = df.groupby([lat, lon]).size().reset_index(name='count')[[lat, lon]]
    df_new['geo-id'] = df_new.index
    df_new['geo-id'] = df_new['geo-id'].map(lambda x: '(' + str(x) + ')')
    return df.merge(df_new, how='left', on=[lat, lon])


def pre_processing(df_raw, min_records=150, freq='H',
                   filesave=False, geoid=False, resolution=4):
    """ pre-processing the given dataset
    :param freq: default 'H', we can use 'D' for day, 'M' for month, 'W' for week
    :param df_raw: dataframe, raw dataset
    :param min_records: the min requirement of users' records, remove all invalid users' information.
    :param filesave: whether save the pre-processed results
    :param geoid: whether use geo-id
    :param resolution: when geo-id used, the resolution
    :return: pre-processed dataframe
    """
    df_wp = df_raw.dropna(subset=['userid', 'placeid', 'datetime'])[[
        'userid', 'placeid', 'datetime', 'lat', 'lon']]
    # for weeplace dataset, '-' also means missing placeid
    df_wp = df_wp[df_wp['placeid'] != '-']

    df = df_wp.groupby('userid')['datetime'].count().reset_index(name='count')
    mask1 = df['count'].values >= min_records
    user = pd.DataFrame(df.values[mask1], df.index[mask1], df.columns)['userid'].tolist()
    # for computation, ignore minutes and seconds (hourly-based)
    # we still have many choice, day, week, month level
    df_wp['datetime'] = pd.to_datetime(df_wp['datetime'])
    df_wp['datetimeH'] = pd.to_datetime(df_wp['datetime']).dt.floor(freq)

    df_processed = df_wp[df_wp['userid'].isin(user)]

    if geoid:
        df_processed = geo2id(df_processed, resolution)
        df_processed = df_processed.drop('placeid', axis=1)
        df_processed = df_processed.rename(columns={'geo-id': 'placeid'})

    df_processed = df_processed.drop(['lat', 'lon'], axis=1)

    if filesave:
        if geoid:
            name = 'data/weeplace_checkins_' + str(min_records) + 'Geo-UPD' + str(resolution) + '.csv'
        else:
            name = 'data/weeplace_checkins_' + str(min_records) + 'UPD.csv'
        df_processed.to_csv(name, index=False)

    return df_processed

