#!/usr/bin/env python
"""
Download the data required for the data science 109 project
for Sam, Luke, Jake, and Jeremy's group.
"""

import datetime
import optparse
import requests
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sys import exit

SOCRATA_TOKEN='jWNb7dvQQlhOyTJPutAIGJheR'

INSPECTION_ENDPOINT = 'https://data.cityofchicago.org/resource/cwig-ma7x.json'
BUSINESS_ENDPOINT = 'https://data.cityofchicago.org/resource/xqx5-8hwx.json'
CRIME_ENDPOINT = 'https://data.cityofchicago.org/resource/6zsd-86xi.json'
SANITATION_ENDPOINT = 'https://data.cityofchicago.org/resource/kcdz-f29q.json'

def make_request(url, auth=SOCRATA_TOKEN):
    """
    Issue a token-authed request to the required URL.
    """
    if auth is SOCRATA_TOKEN:
        return requests.get(url, headers={'X-App-Token': SOCRATA_TOKEN})
    elif auth is NOAA_TOKEN:
        return requests.get(url, headers={'token': NOAA_TOKEN})

# def build_weather_query(location_zip, start, end):
#     """
#     Build the query URL for the inspections data. 
#     """

#     return "{}&locationid=ZIP:{}&startdate={}&enddate={}".format(NOAA_BASE, location_zip, start.isoformat(), end.isoformat())

def build_inspection_query(start):
    """
    Given an inspection date, build a query for all of the inspections
    occuring on or after that day.
    """
    return "{}?$where=inspection_date>='{}'&$limit=10000000".format(INSPECTION_ENDPOINT, start.isoformat())

def build_crime_query(start):
    """
    Build the URL query to retrieve crime data.
    """
    return "{}?$where=date>='{}'&$limit=10000000".format(CRIME_ENDPOINT, start.isoformat())

def build_business_query(ids):
    """
    Build the URL query for the license numbers
    """
    # in_set = ", ".join([repr(str(i)) for i in biz_ids])
    return "{}?$where=license_number in({})&$limit=10000000".format(BUSINESS_ENDPOINT, ', '.join(["'{}'".format(i) for i in ids]))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def get_business_data(ids):
    """
    Retrieve business license record data for all of the ids in ids
    batching requests as appropriate (can request ~30 at a time)
    """
    urls = [build_business_query(id_chunk) for id_chunk in chunks(ids, 300)]

    json_data = []
    for url in urls:
        try:
            try:
                data = make_request(url).json()
                json_data.extend(data)
                print "Succeeded!"
            except requests.HTTPError as e:
                print e
                continue
        except Exception as e:
            print e
            print type(data)
            continue

    return json_data

def build_sanitation_query(start):
    """
    Build the query URL for sanitation data in the period of interest
    """
    return "{}?$where=creation_date>='{}'&$limit=10000000".format(SANITATION_ENDPOINT, start.isoformat())

def get_loc(row):
    return row['latitude'], row['longitude']

def distance(d1, d2):
    lat1, lon1 = d1
    lat2, lon2 = d2
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

def count_incidents_nearby(row, incident_df, num_days = 5, max_dist=5):
    def get_loc(r):
        return r['latitude'], r['longitude']

    d1 = get_loc(row)
    
    def keep(d):
        recent = abs((d['creation_date'] - row['inspection_date']).days) < num_days
        before = abs((d['creation_date'] - row['inspection_date']).days) >= 0
        return (recent and before)
    
    last_n_days = incident_df.apply(lambda d: keep(d),axis=1 )
    valid =incident_df[last_n_days]
    
    if len(valid) < 1:
        return 0
    
    dists = valid.apply(lambda r: distance(d1, get_loc(r)), axis=1) < max_dist
    
    return len(valid[dists])

def count_crimes_nearby(row, incident_df, num_days = 5, max_dist=5):
    def get_loc(r):
        return r['latitude'], r['longitude']

    d1 = get_loc(row)
    
    def keep(d):
        recent = abs((d['date'] - row['inspection_date']).days) < num_days
        before = abs((d['date'] - row['inspection_date']).days) >= 0
        return (recent and before)
    
    last_n_days = incident_df.apply(lambda d: keep(d),axis=1 )
    valid =incident_df[last_n_days]
    
    if len(valid) < 1:
        return 0
    
    dists = valid.apply(lambda r: distance(d1, get_loc(r)), axis=1) < max_dist
    
    return len(valid[dists])

def filter_sanitation_police(df):
    idate = df['inspection_date']
    cdate = df['creation_date']
    
    before = cdate < idate
    import datetime
    after = abs(cdate - idate) < datetime.timedelta(days=5)
    
    return df[before & after]

def filter_crime_by_police(df):
    idate = df['date']
    cdate = df['creation_date']
    
    before = cdate < idate
    import datetime
    after = abs(cdate - idate) < datetime.timedelta(days=5)
    
    ret = df[before & after].copy()
    del ret['date']
    return ret

def main(date=datetime.datetime.today(),export=True):
    """
    Runner function.
    """
    start_date = date
    
    print "Fetching data beginning {}.".format(start_date.date())

    insp_req = build_inspection_query(start_date)

    try:
        inspection_data = make_request(insp_req).json()
    except requests.HTTPError as excep:
        print excep
        print "Inspection data retrieval failed."
        exit(1)

    if len(inspection_data) < 1:
        print "No inspection data received."
        exit(1)
    else:
        print "{} records received. Finding businesses...".format(len(inspection_data))

    ids = list(set([str(d.get('license_')) for d in inspection_data]))

    business_data = get_business_data(ids)

    print "RECEIVED {} business details".format(len(business_data))
    
    
    print inspection_data[0].keys()
    inspection_df = pd.DataFrame(inspection_data).loc[:, [u'dba_name',
                                                          u'facility_type',
                                                          u'inspection_date',
                                                          u'inspection_type',
                                                          u'insepction_id',
                                                          u'violations',
                                                          u'latitude',
                                                          u'license_',
                                                          u'longitude',
                                                          u'results',
                                                          u'risk',
                                                          u'zip']]
    bad_indices = []
    good_indices = []
    for biz_index in range(len(business_data)):
        biz_dict = business_data[biz_index]
        if type(biz_dict) != type({1: 0}):
            bad_indices.append(biz_index)
        else:
            good_indices.append(biz_index)
    
    
    business_data = list(np.array(business_data)[good_indices])
    


    biz_df = pd.DataFrame(business_data).loc[:, [u'business_activity', u'date_issued', 
                                                 u'latitude', u'license_description',
                                                 u'license_number', u'license_start_date',
                                                 u'license_status', u'longitude',
                                                 u'police_district', u'precinct',
                                                 u'site_number', u'ward_precinct', u'zip_code']]


    biz_df = biz_df.set_index('license_number')
    biz_df['license_start_date'] = pd.to_datetime(biz_df['license_start_date'])

    biz_df = biz_df.groupby(biz_df.index).apply(lambda g: g[g['license_start_date'] == g['license_start_date'].max()])
    biz_df.index = biz_df.index.droplevel()

    insp_biz_df = inspection_df.join(biz_df, rsuffix='_biz', on='license_')
    
    insp_biz_df['latitude'] = pd.to_numeric(insp_biz_df['latitude'])
    insp_biz_df['longitude'] = pd.to_numeric(insp_biz_df['longitude'])
    insp_biz_df['latitude_biz'] = pd.to_numeric(insp_biz_df['latitude_biz'],errors='coerce')
    insp_biz_df['longitude_biz'] = pd.to_numeric(insp_biz_df['longitude_biz'],errors='coerce')
    insp_biz_df['inspection_date'] = pd.to_datetime(insp_biz_df['inspection_date'])

    sanitation_url = build_sanitation_query(start_date)
    try:
        sanitation_data = make_request(sanitation_url).json()
    except requests.HTTPError as excep:
        print crime_url
        print excep
        exit(1)
    print "SANITATION LEN: {}".format(len(sanitation_data))

    sanitation_df = pd.DataFrame(sanitation_data).loc[:, [
                                                     u'creation_date',
                                                     u'police_district',
                                                     u'latitude',
                                                     u'longitude',
                                                     u'zip_code']]
    sanitation_df['latitude'] = pd.to_numeric(sanitation_df['latitude'])
    sanitation_df['longitude'] = pd.to_numeric(sanitation_df['longitude'])
    sanitation_df['creation_date'] = pd.to_datetime(sanitation_df['creation_date'])

    sanit_df = pd.DataFrame(sanitation_df).groupby(['creation_date','police_district']).count().reset_index().set_index('police_district')
    del sanit_df['latitude']
    del sanit_df['longitude']
    sanit_df.columns.values[1] = 'num_sanitation_in_police_dist'
    sanit_df = sanit_df.reset_index()

    ###
    ### This code works, but is also very slow. Only run it if you find out the dataset has a lot of 
    ### missing values. 
    # insp_biz_df['sanitation_by_location'] = insp_biz_df.apply(lambda r: count_incidents_nearby(r, sanitation_df),axis=1)
    # insp_biz_df = insp_biz_df.join(sanit_df, on=['police_district'])
    insp_biz_df = pd.merge(insp_biz_df, sanit_df, left_on=['inspection_date', 'police_district'], right_on=['creation_date', 'police_district'], how='left')

    crime_url = build_crime_query(start_date)
    try:
        crime_data = make_request(crime_url).json()
    except requests.HTTPError as excep:
        print crime_url
        print excep
        exit(1)
    print "CRIME LEN: {}".format(len(crime_data))

    crime_df = pd.DataFrame(crime_data).loc[:, ['date', 'latitude', 'longitude', 'district']]
    crime_df['date'] = pd.to_datetime(crime_df['date'])
    crime_df['district'] = pd.to_numeric(crime_df['district'])
    crime_df['latitude'] = pd.to_numeric(crime_df['latitude'])
    crime_df['longitude'] = pd.to_numeric(crime_df['longitude'])

    time = pd.DatetimeIndex(crime_df['date'])
    crime_by_district = crime_df.groupby([crime_df['district'],time.date]).count()
    crime_by_district['crime_count_by_district'] = crime_by_district['latitude']
    del crime_by_district['latitude']
    del crime_by_district['longitude']
    del crime_by_district['date']
    del crime_by_district['district']

    crime_by_district = crime_by_district.reset_index()

    crime_by_district['date'] = crime_by_district['level_1']
    del crime_by_district['level_1']
    crime_by_district['date'] = pd.to_datetime(crime_by_district['date'])
    crime_by_district = crime_by_district.reset_index()

    insp_biz_df['police_district'] = pd.to_numeric(insp_biz_df['police_district'])
    insp_biz_df = pd.merge(insp_biz_df, crime_by_district, left_on=['inspection_date', 'police_district'], right_on=['date', 'district'], how='left')

    ###
    ### This code works, but it's really really fucking slow, so maybe don't run it.
    ###
    #insp_biz_df['crimes_by_location'] = insp_biz_df.apply(lambda r: count_crimes_nearby(r, crime_df),axis=1)
    import os
    if export == True:
        insp_biz_df.to_csv('DOWNLOADED_DATA.csv', index=False)
    
    # weather_url = build_weather_query('02138', start_date,
    #                                        datetime.datetime.today())
    # print weather_url
    # print make_request(,
    #                    auth=NOAA_TOKEN).json()

                 auth=NOAA_TOKEN).json()


if __name__ == '__main__':
    if SOCRATA_TOKEN is "":
        print "Token missing (see file and talk to Sam for required.)"
        exit(1)

    parser = optparse.OptionParser()
    parser.add_option('-d', '--start_date', dest='start_date',
                      help='specify the start date for inspection collections.')

    opts, _ = parser.parse_args()

    # build the request
    if opts.start_date is None:
        main()
    else:
        main(date=datetime.datetime.strptime(opts.start_date, '%m/%d/%Y'))