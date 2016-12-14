#!/usr/bin/env python
"""
Download crime and sanitation date, 
split to buckets and export bucket counts.
"""
from download import make_request, CRIME_ENDPOINT, SANITATION_ENDPOINT
import datetime
import pandas as pd
import numpy as np
import requests
import itertools

def build_crime_query(start, end):
    return "{}?$where=date>='{}' AND date<='{}'&$limit=10000000".format(CRIME_ENDPOINT, start.isoformat(), end.isoformat())

def build_sanitation_query(start, end):
    return "{}?$where=creation_date>='{}' AND creation_date<='{}'&$limit=10000000".format(SANITATION_ENDPOINT, start.isoformat(), end.isoformat())

def download_crime(start_date, end_date):
    crime_url = build_crime_query(start_date, end_date)

    print crime_url

    try:
        crime_df = pd.DataFrame(make_request(crime_url).json())
    except requests.HTTPError as excep:
        print crime_url
        print excep
        exit(1)

    print "Crime records received: {}".format(crime_df.shape[0])

    crime_df = crime_df.loc[:, ['date', 'latitude', 'longitude']]
    crime_df['date'] = pd.to_datetime(crime_df['date'])
    crime_df['latitude'] = pd.to_numeric(crime_df['latitude'])
    crime_df['longitude'] = pd.to_numeric(crime_df['longitude'])

    return crime_df

def download_sanit(start, end):
    sanit_url = build_sanitation_query(start, end)
    print sanit_url

    try:
        sanitation_df = pd.DataFrame(make_request(sanit_url).json())
    except requests.HTTPError as excep:
        print sanit_url
        print excep
        exit(1)
    print "Sanitation records received: {}".format(sanitation_df.shape[0])

    sanitation_df = sanitation_df.loc[:, [
                                                     u'creation_date',
                                                     u'police_district',
                                                     u'latitude',
                                                     u'longitude',
                                                     u'zip_code']]
    sanitation_df['latitude'] = pd.to_numeric(sanitation_df['latitude'])
    sanitation_df['longitude'] = pd.to_numeric(sanitation_df['longitude'])
    sanitation_df['creation_date'] = pd.to_datetime(sanitation_df['creation_date'])

    return sanitation_df

def build_grid(lat, lon, bins_x):

    bins_y = (float((lon[1] - lon[0]))/float((lat[1] - lat[0])))*bins_x
    
    lat_points = np.linspace(lat[0], lat[1], bins_x + 2)
    lon_points = np.linspace(lon[0], lon[1], bins_y + 2)
    
    grid = list(itertools.product(lon_points, lat_points))
    
    lat_space = lat_points[1] - lat_points[0]
    lon_space = lon_points[1] - lon_points[0]
    
    return lat_space, lon_space, grid

def subset_count(df, center, lat_space, lon_space):
    lonc, latc = center
    
    min_lat = latc - lat_space
    max_lat = latc + lat_space
    
    min_lon = lonc - lon_space
    max_lon = lonc + lon_space
    
    in_lat = (min_lat <= df['latitude']) & (df['latitude'] <= max_lat)
    in_lon = (min_lon <= df['longitude']) & (df['longitude'] <= max_lon)
    
    return len(df[in_lat & in_lon])


def main(start_date, end_date, n_points):    
    
    crime_df = download_crime(start_date, end_date)
    sanit_df = download_sanit(start_date, end_date)

    return crime_df, sanit_df
