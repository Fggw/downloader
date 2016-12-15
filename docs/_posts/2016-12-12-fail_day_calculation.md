---
layout: post
title:  "Fail Day Calculation"
published: true
page: 8
---


```python
from download import main
```

First we load in the data:


```python
import datetime
import optparse
import requests
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sys import exit
import math
import matplotlib.pyplot as plt
import seaborn

%matplotlib inline

data = pd.read_csv('DOWNLOADED_DATA.csv')


```


```python

result_possibilities = ['Pass', 'Fail', 'Pass w/ Conditions']
valid_result_bools = [result in result_possibilities for result in data.results]
data = data[valid_result_bools]
```


```python
def fail(result):
    if 'Fail' in result:
        return 1
    else:
        return 0
    
    
def grocery_store(facility_type):
    if 'Grocery Store' == facility_type:
        return 0
    else:
        return 1
```


```python
data['result_binary'] = [fail(result) for result in data.results]
data['grocery'] = [grocery_store(facility_type) for facility_type in data.facility_type]
```


```python
top_inspection_types = data.groupby('inspection_type').count().result_binary.sort_values(ascending=False).index[:10]
top_inspection_types_sorted = data.groupby('inspection_type').mean().result_binary[top_inspection_types].sort_values(ascending=False).index

fig = plt.figure(figsize=(20,6)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
width = 0.4

plt.rc('font', family='serif')
plt.rc('font', family='serif')


data.groupby('inspection_type').mean().result_binary[top_inspection_types].sort_values(ascending=False).plot(kind='bar',color='r',ax = ax, alpha=0.7,width=width,position=0)
percentage_1 = data.groupby('inspection_type').count().result_binary[top_inspection_types_sorted]/len(data)
percentage_1.plot(kind='bar',color='b',ax = ax2, alpha=0.7,width=width,position=1)
ax.set_ylabel('Failure Rate',fontsize = 20)
ax2.set_ylabel('Fraction of Total Inpsections',fontsize = 20)
ax.set_xlabel('Inspection Type',fontsize = 20)
plt.title('Failure Rates by Inspection Type',fontsize = 32)
ax.set_xticklabels(labels = top_inspection_types_sorted, ha = 'right')
leg = ax.legend(['Failure Rate'],loc=1, bbox_to_anchor=(0.825, 0.95),fontsize = 20)
leg2 = ax2.legend(['Fraction of Total Inspections'],loc=1, bbox_to_anchor=(0.975, 0.825),fontsize = 20)
ax.set_xlim(-0.6,9.6)

leg.get_frame().set_linewidth(0.0)
leg2.get_frame().set_linewidth(0.0)


for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                tick.label.set_rotation(30)

              
                
for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 

for label2 in ax2.yaxis.get_majorticklabels():
                label2.set_fontsize(15) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')  

```


![png](fail_day_calculation_files/fail_day_calculation_7_0.png)



```python
inspection_type_dummies = pd.get_dummies(data['inspection_type'])
inspection_type_dummies = inspection_type_dummies[top_inspection_types]


data = pd.concat([data,inspection_type_dummies],axis=1,join='inner')


biz_cols = []
for col in data.columns:
    if 'business_activity' in col:
        biz_cols.append(col)
        
count = []
mean = []
for col in biz_cols:
    mean.append(data[data[col] == 1].result_binary.mean())
    count.append(len(data[data[col] == 1].result_binary))
```


```python
biz_cols_mean = pd.Series(mean,biz_cols)
biz_cols_count = pd.Series(count,biz_cols)

biz_cols_count_top = biz_cols_count.sort_values(ascending=False).index[:10]


biz_col_mean_sorted = biz_cols_mean[biz_cols_count_top].sort_values(ascending=False).index

fig = plt.figure(figsize = (20,6)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

biz_cols_mean[biz_col_mean_sorted].sort_values(ascending=False).plot(kind='bar',color='r',ax = ax, alpha=0.7,width=width,position=0)
percentage = biz_cols_count[biz_col_mean_sorted]/len(data)
percentage.plot(kind='bar',color='b',ax = ax2, alpha=0.7,width=width,position=1)



ax.set_ylabel('Failure Rate',fontsize = 20)
ax2.set_ylabel('Fraction of Total Inpsections',fontsize = 20)
ax.set_xlabel('Inspection Type',fontsize = 20)
plt.title('Failure Rates by Most Common Business Activities',fontsize = 32)
leg = ax.legend(['Failure Rate'],loc=1, bbox_to_anchor=(0.6, 1),fontsize = 20)
leg2 = ax2.legend(['Fraction of Total Inspections'],loc=1, bbox_to_anchor=(0.75, 0.9),fontsize = 20)
ax.set_xlim(-0.6,9.6)

leg.get_frame().set_linewidth(0.0)
leg2.get_frame().set_linewidth(0.0)



ax.xaxis.set_ticklabels(['Tavern',
 'Sale of Tobacco', 
 'Onsite Entertainment', 
 'Food & Seating',
 'Liquor Outdoors',
 'Consumption of Liquor on Premise',
 'Sales, Packaged Liquor',
 'Food & Dining Area',
 'Sales, Perishable Foods',
 'Catering of Liquor'],fontsize = 20, ha = 'right')


for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                tick.label.set_rotation(30)

              
                
for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 

for label2 in ax2.yaxis.get_majorticklabels():
                label2.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')  
plt.show()
```


![png](fail_day_calculation_files/fail_day_calculation_9_0.png)



```python
results_data = pd.read_csv('pred_prob_for_fails_cv.csv')
results_data['inspection_date'] = pd.to_datetime(results_data['inspection_date'],format = "%y-%m-%d")
results_data['ids'] = results_data['Unnamed: 0']


```


```python
results_data.head()


```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>AdaBoost, log loss: 0.73, params = ['n_estimators: 100']</th>
      <th>KNN, log loss: 17.24, params = ['n_neighbors: 50']</th>
      <th>LDA, log loss: 0.72, params = ['shrinkage: 0.0', 'solver: lsqr']</th>
      <th>LogReg, log loss: 0.75, params = ['C: 10.0']</th>
      <th>QDA, log loss: 1.07, params = ['reg_param: 0.25']</th>
      <th>RandomForest, log loss: 0.96, params = ['n_estimators: 50', 'max_features: 45', 'max_depth: 2']</th>
      <th>inspection_date</th>
      <th>result_binary</th>
      <th>ids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>0.500691</td>
      <td>0.42</td>
      <td>0.517646</td>
      <td>0.522678</td>
      <td>0.472780</td>
      <td>0.475355</td>
      <td>2016-01-26</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>0.493252</td>
      <td>0.40</td>
      <td>0.101334</td>
      <td>0.090216</td>
      <td>0.163932</td>
      <td>0.259399</td>
      <td>2016-06-23</td>
      <td>0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29</td>
      <td>0.498136</td>
      <td>0.42</td>
      <td>0.383257</td>
      <td>0.378739</td>
      <td>0.372917</td>
      <td>0.475355</td>
      <td>2016-03-01</td>
      <td>0</td>
      <td>29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>0.498669</td>
      <td>0.48</td>
      <td>0.478759</td>
      <td>0.466773</td>
      <td>0.516154</td>
      <td>0.475355</td>
      <td>2016-09-16</td>
      <td>0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>0.499631</td>
      <td>0.36</td>
      <td>0.488194</td>
      <td>0.498767</td>
      <td>0.779546</td>
      <td>0.475355</td>
      <td>2016-06-29</td>
      <td>0</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>




```python

from dateutil.relativedelta import relativedelta  
import seaborn
seaborn.set_style(style='white')



def list_rank(true_values,probabilities,data_test):
    result_series = pd.Series(probabilities,true_values).sort_values(ascending=False)
    result_time_series = pd.Series(result_series.index,data_test.inspection_date)
    return result_series,result_time_series
    

def series_running(result_time_series):
    results = np.array(result_time_series.result_binary)
    running_mean = []
    for index in range(len(results)-1):
        running_mean.append(np.mean(results[0:index+1]))
    return pd.Series(running_mean,result_time_series.index[1:])



def fail_days(result_timeseries):
    date_array = np.array(result_timeseries.index)
    result_array = np.array(result_timeseries.result_binary)
    start_date = pd.to_datetime(date_array[0]) - relativedelta(days=1)
    fail_days = 0
    for date_ind in range(len(result_timeseries.index)):
        date = date_array[date_ind]
        delta = pd.to_datetime(date) - start_date
        fail_days += delta.days*result_array[date_ind]
    return fail_days



```


```python
test_columns = results_data.columns[-3:]
model_columns = results_data.columns[1:-3]
true_values = results_data[results_data.columns[-1]]
test_data = results_data[test_columns].sort_values('inspection_date')
test_data = test_data.set_index([test_data.inspection_date])

```


```python
fail_day_count_mean = []
for model_column in model_columns:
    result_series = results_data[[model_column,'result_binary','ids']]
    sorted_series = result_series.sort_values(result_series.columns[0],ascending=False)
    sorted_series = sorted_series.set_index([test_data.inspection_date])
    fail_day_count_mean.append(fail_days(sorted_series)/test_data.result_binary.sum())
    
    
    
```


```python
true_fail_days = fail_days(test_data)/test_data.result_binary.sum()



```


```python
sorted_failday_results
```




    Actual                                                                                             152
    KNN, log loss: 17.24, params = ['n_neighbors: 50']                                                 149
    RandomForest, log loss: 0.96, params = ['n_estimators: 50', 'max_features: 45', 'max_depth: 2']    120
    QDA, log loss: 1.07, params = ['reg_param: 0.25']                                                  119
    LDA, log loss: 0.72, params = ['shrinkage: 0.0', 'solver: lsqr']                                   112
    LogReg, log loss: 0.75, params = ['C: 10.0']                                                       111
    AdaBoost, log loss: 0.73, params = ['n_estimators: 100']                                           108
    dtype: int64




```python
all_cols = list(model_columns)
all_cols.append('Actual')
color_series = pd.Series(['w']*len(model_columns) + ['r'],index = all_cols)
sorted_failday_results = pd.Series(fail_day_count_mean + [true_fail_days],index = all_cols).sort_values(ascending=False)

color_list =  list(color_series[sorted_failday_results.index])



fig = plt.figure(figsize=(20,12))
ax1 = fig.add_subplot(111) # Create matplotlib axes

sorted_failday_results.plot(kind = 'bar',color= color_list,ax = ax1, edgecolor='black',lw=1)


plt.title('Average Days Until Inspection',fontsize=32)
plt.rc('font', family='serif')
plt.xlabel('Model',fontsize=20)
plt.ylabel('Fail Days',fontsize=20)
ax1.set_xticklabels(labels = list(sorted_failday_results.index), ha = 'right')

ax1.xaxis.set_ticklabels(['Actual',
 'KNN', 
 'Random Forest', 
 'QDA',
 'LDA',
 'LogReg',
 'AdaBoost'],fontsize = 20, ha = 'right')




for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                tick.label.set_rotation(45)
                
                
for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 
                
                
```


![png](fail_day_calculation_files/fail_day_calculation_17_0.png)



```python
fig = plt.figure(figsize=(20,10))
ax3 = fig.add_subplot(111) # Create matplotlib axes
ax4 = ax3.twinx() # Create another axes that shares the same x-axis as ax.

test_data.groupby('inspection_date').count().plot(ax = ax3)
test_data.groupby('inspection_date').mean()['result_binary'].plot(ax = ax4,color='r')

print test_data.groupby('inspection_date').count().mean()


ax3.legend(['Number of Inspections'],loc=1, bbox_to_anchor=(1, 1),fontsize = 20)
ax4.legend(['Failure Rate'],loc=1, bbox_to_anchor=(0.905, 0.925),fontsize = 20)

plt.rc('font', family='serif')
plt.title('Daily Inspection Number and Failure Rate',fontsize=32)
plt.xlabel('Date',fontsize=20)
ax3.set_ylabel('Number of Inspections',fontsize = 20)
ax4.set_ylabel('Failure Rate',fontsize = 20)

for tick in ax3.xaxis.get_major_ticks():
                tick.label.set_fontsize(10) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 
                
                
for tick in ax4.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 

            
for label3 in ax3.yaxis.get_majorticklabels():
                label3.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 
                
for label4 in ax4.yaxis.get_majorticklabels():
                label4.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')  
                
```

    result_binary    69.04386
    ids              69.04386
    dtype: float64



![png](fail_day_calculation_files/fail_day_calculation_18_1.png)



```python

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(111) # Create matplotlib axes

for model_column in model_columns:
    result_series = results_data[[model_column,'result_binary']]
    sorted_series = result_series.sort_values(result_series.columns[0],ascending=False)
    sorted_series = sorted_series.set_index([test_data.inspection_date])
    series_running(sorted_series).plot(ax = ax1)
    
series_running(test_data).plot(legend = True, ax = ax1,color='r')
plt.legend(list(model_columns),loc=1, bbox_to_anchor=(0.9, 0.95),fontsize = 15)
plt.title('Cumulative Failure Rate For Model Ordering',fontsize=20)
plt.xlabel('Time of Inspection',fontsize=15)
plt.ylabel('Failure Rate',fontsize=15)

```




    <matplotlib.text.Text at 0x10dfdad10>




![png](fail_day_calculation_files/fail_day_calculation_19_1.png)



```python
model_column = model_columns[0]
result_series = results_data[[model_column,'result_binary']]
sorted_series = result_series.sort_values(result_series.columns[0],ascending=False)
sorted_series = sorted_series.set_index([test_data.inspection_date])
result_data_rolling_mean = sorted_series.result_binary.rolling(window = 1000).mean()
result_data_rolling_mean.plot(ax = ax1, legend= True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x140b1b190>




```python

```


```python
list(model_columns)
```




    ["AdaBoost, log loss: 0.73, params = ['n_estimators: 100']",
     "KNN, log loss: 17.24, params = ['n_neighbors: 50']",
     "LDA, log loss: 0.72, params = ['shrinkage: 0.0', 'solver: lsqr']",
     "LogReg, log loss: 0.75, params = ['C: 10.0']",
     "QDA, log loss: 1.07, params = ['reg_param: 0.25']",
     "RandomForest, log loss: 0.96, params = ['n_estimators: 50', 'max_features: 45', 'max_depth: 2']"]




```python

fig = plt.figure(figsize=(20,12))
ax1 = fig.add_subplot(111) # Create matplotlib axes

for model_column in model_columns:
    result_series = results_data[[model_column,'result_binary']]
    sorted_series = result_series.sort_values(result_series.columns[0],ascending=False)
    sorted_series = sorted_series.set_index([test_data.inspection_date])
    result_data_rolling_mean = sorted_series.result_binary.rolling(window = 1000).mean()
    result_data_rolling_mean.plot(ax = ax1, legend= True)
    
test_data.result_binary.rolling(window =1000).mean().plot(legend = True, color = 'r',ax = ax1)

plt.legend(['AdaBoost',
 'KNN', 
 'LDA',
 'LogReg',
 'QDA',
 'Random Forest',
 'Actual'],loc=1, fontsize = 20)
plt.rc('font', family='serif')
plt.title('Rolling Inspection Failure Rate For Best Model Rank (n = 1000)',fontsize=32)
plt.xlabel('Time of Inspection',fontsize=20)
plt.ylabel('Inspection Failure Rate',fontsize=20)
for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 
for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 



```


![png](fail_day_calculation_files/fail_day_calculation_23_0.png)



```python
rest_data_rolling_mean = pd.rolling_mean(test_data_series,1000).plot()


```


```python

```


```python
all_ada_result = results_data[[model_columns[0],'result_binary','ids','inspection_date']]
all_ada_result = all_ada_result.set_index([all_ada_result.ids])
all_ada_result = all_ada_result.sort_values('inspection_date')
all_ada_result.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AdaBoost, log loss: 0.73, params = ['n_estimators: 100']</th>
      <th>result_binary</th>
      <th>ids</th>
      <th>inspection_date</th>
    </tr>
    <tr>
      <th>ids</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1893</th>
      <td>0.497859</td>
      <td>0</td>
      <td>1893</td>
      <td>2016-01-04</td>
    </tr>
    <tr>
      <th>71285</th>
      <td>0.499958</td>
      <td>1</td>
      <td>71285</td>
      <td>2016-01-05</td>
    </tr>
    <tr>
      <th>81062</th>
      <td>0.501633</td>
      <td>0</td>
      <td>81062</td>
      <td>2016-01-05</td>
    </tr>
    <tr>
      <th>51941</th>
      <td>0.495638</td>
      <td>0</td>
      <td>51941</td>
      <td>2016-01-05</td>
    </tr>
    <tr>
      <th>116038</th>
      <td>0.500380</td>
      <td>0</td>
      <td>116038</td>
      <td>2016-01-05</td>
    </tr>
  </tbody>
</table>
</div>




```python

day_per_period = 30

#arbitrary
days_of_inspections_added = 30

#arbitarily decreased to make it so that more inspections come in than come out
inspections_per_day = 20
claim_rate_vector = []


```


```python
def fail_days(result_timeseries,start_date):
    date_array = np.array(result_timeseries.index)
    result_array = np.array(result_timeseries.result_binary)
    fail_days = 0
    for date_ind in range(len(result_timeseries.index)):
        date = date_array[date_ind]
        delta = pd.to_datetime(date) - start_date
        fail_days += delta.days*result_array[date_ind]
    return fail_days
```


```python


remaining_ada_result = all_ada_result
inspected_ids_all = []
start_date = all_ada_result.inspection_date.iloc[0]
fail_days_total = []
fail_count_total = []
for period in range(int(20) + 1):
    
    current_date = all_ada_result.inspection_date.iloc[0] + relativedelta(days = day_per_period)*(period)
    last_visible_date = all_ada_result.inspection_date.iloc[0] + relativedelta(days = day_per_period)*(period+1) + relativedelta(months = 0)
    
    visible_ada_result = remaining_ada_result[remaining_ada_result.inspection_date <= last_visible_date]
    sorted_series = visible_ada_result.sort_values(visible_ada_result.columns[0],ascending=False)

    day_delta = np.array([inspection_num/(inspections_per_day*day_per_period) for inspection_num in np.array(range(len(sorted_series)))+1])
    date_inspected = current_date + day_delta*relativedelta(days=1)
    
    sorted_series = sorted_series.set_index([date_inspected])
    
    
    
    
 
    inspected_ids = sorted_series.ids.iloc[0:inspections_per_day*day_per_period]
    inspected_series = sorted_series.iloc[0:inspections_per_day*day_per_period]
    fail_days_total.append(fail_days(inspected_series,start_date))
    fail_count_total.append(inspected_series.result_binary.sum())

    
    remaining_ada_result = remaining_ada_result.drop(np.array(inspected_ids))
    
    #plot running mean of inspected restaurants
    inspected_series.result_binary.rolling(window = 419).mean().plot()
    
print inspected_series.inspection_date.iloc[-1] - start_date
    
    
```

    71 days 00:00:00



![png](fail_day_calculation_files/fail_day_calculation_29_1.png)



```python
fail_count_total
```




    [173,
     208,
     199,
     196,
     187,
     220,
     188,
     206,
     206,
     175,
     177,
     144,
     126,
     123,
     128,
     126,
     128,
     110,
     99,
     96,
     77]




```python
np.sum(fail_days_total)/np.sum(np.array(fail_count_total))
```




    255




```python
fail_days_total
```




    [0,
     6240,
     11940,
     17641,
     22440,
     33001,
     33840,
     43260,
     49440,
     47250,
     53100,
     47520,
     45360,
     47970,
     53761,
     56700,
     61440,
     56100,
     53460,
     54721,
     46200]




```python
inspected_series
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AdaBoost, log loss: 0.73, params = ['n_estimators: 100']</th>
      <th>result_binary</th>
      <th>ids</th>
      <th>inspection_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-12-05</th>
      <td>0.498905</td>
      <td>0</td>
      <td>91994</td>
      <td>2016-09-14</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498905</td>
      <td>0</td>
      <td>90976</td>
      <td>2016-09-27</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498905</td>
      <td>0</td>
      <td>89523</td>
      <td>2016-09-13</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498905</td>
      <td>0</td>
      <td>76327</td>
      <td>2016-09-13</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498905</td>
      <td>0</td>
      <td>95772</td>
      <td>2016-09-16</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498904</td>
      <td>0</td>
      <td>89905</td>
      <td>2016-05-16</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498904</td>
      <td>1</td>
      <td>80697</td>
      <td>2016-05-13</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498903</td>
      <td>0</td>
      <td>12688</td>
      <td>2016-09-30</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498903</td>
      <td>0</td>
      <td>70981</td>
      <td>2016-04-28</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498902</td>
      <td>0</td>
      <td>11246</td>
      <td>2016-11-04</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498902</td>
      <td>1</td>
      <td>23977</td>
      <td>2016-02-22</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498901</td>
      <td>0</td>
      <td>17388</td>
      <td>2016-02-19</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498901</td>
      <td>1</td>
      <td>27462</td>
      <td>2016-05-16</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498901</td>
      <td>1</td>
      <td>21682</td>
      <td>2016-08-24</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498900</td>
      <td>0</td>
      <td>40839</td>
      <td>2016-07-26</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498900</td>
      <td>0</td>
      <td>9532</td>
      <td>2016-07-18</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498900</td>
      <td>0</td>
      <td>11798</td>
      <td>2016-07-28</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498900</td>
      <td>1</td>
      <td>6592</td>
      <td>2016-05-06</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498900</td>
      <td>1</td>
      <td>33280</td>
      <td>2016-09-07</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498899</td>
      <td>0</td>
      <td>80157</td>
      <td>2016-03-25</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498899</td>
      <td>1</td>
      <td>35298</td>
      <td>2016-06-09</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498898</td>
      <td>1</td>
      <td>61805</td>
      <td>2016-09-02</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498898</td>
      <td>0</td>
      <td>374</td>
      <td>2016-06-29</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498896</td>
      <td>1</td>
      <td>20601</td>
      <td>2016-10-25</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498895</td>
      <td>0</td>
      <td>24434</td>
      <td>2016-05-09</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498895</td>
      <td>0</td>
      <td>12600</td>
      <td>2016-10-27</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498894</td>
      <td>0</td>
      <td>18837</td>
      <td>2016-07-28</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498894</td>
      <td>1</td>
      <td>41870</td>
      <td>2016-07-13</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498894</td>
      <td>0</td>
      <td>26515</td>
      <td>2016-03-10</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498893</td>
      <td>0</td>
      <td>107144</td>
      <td>2016-02-25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498778</td>
      <td>1</td>
      <td>22190</td>
      <td>2016-07-26</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498777</td>
      <td>1</td>
      <td>12449</td>
      <td>2016-01-15</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498777</td>
      <td>0</td>
      <td>79414</td>
      <td>2016-05-12</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498777</td>
      <td>0</td>
      <td>61014</td>
      <td>2016-03-30</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498777</td>
      <td>0</td>
      <td>28381</td>
      <td>2016-09-20</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498777</td>
      <td>1</td>
      <td>29962</td>
      <td>2016-06-30</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498776</td>
      <td>0</td>
      <td>85127</td>
      <td>2016-01-06</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498776</td>
      <td>0</td>
      <td>31207</td>
      <td>2016-06-14</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498775</td>
      <td>0</td>
      <td>41839</td>
      <td>2016-06-13</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498775</td>
      <td>0</td>
      <td>3017</td>
      <td>2016-04-14</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498774</td>
      <td>0</td>
      <td>114648</td>
      <td>2016-07-13</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498774</td>
      <td>1</td>
      <td>101835</td>
      <td>2016-02-08</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498774</td>
      <td>0</td>
      <td>100732</td>
      <td>2016-07-08</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498774</td>
      <td>0</td>
      <td>108362</td>
      <td>2016-03-18</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498773</td>
      <td>0</td>
      <td>16176</td>
      <td>2016-11-16</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498773</td>
      <td>0</td>
      <td>8751</td>
      <td>2016-04-12</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498772</td>
      <td>0</td>
      <td>35145</td>
      <td>2016-02-26</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498772</td>
      <td>0</td>
      <td>90072</td>
      <td>2016-07-08</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498772</td>
      <td>0</td>
      <td>94710</td>
      <td>2016-01-22</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498772</td>
      <td>0</td>
      <td>45920</td>
      <td>2016-05-17</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498771</td>
      <td>0</td>
      <td>15953</td>
      <td>2016-01-14</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498771</td>
      <td>0</td>
      <td>16711</td>
      <td>2016-03-17</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498771</td>
      <td>0</td>
      <td>111060</td>
      <td>2016-07-25</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498771</td>
      <td>0</td>
      <td>91913</td>
      <td>2016-03-15</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498771</td>
      <td>0</td>
      <td>61919</td>
      <td>2016-06-17</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498771</td>
      <td>0</td>
      <td>104422</td>
      <td>2016-01-28</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498771</td>
      <td>1</td>
      <td>51672</td>
      <td>2016-08-15</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498770</td>
      <td>1</td>
      <td>109323</td>
      <td>2016-01-12</td>
    </tr>
    <tr>
      <th>2016-12-05</th>
      <td>0.498770</td>
      <td>1</td>
      <td>37286</td>
      <td>2016-07-15</td>
    </tr>
    <tr>
      <th>2016-12-06</th>
      <td>0.498769</td>
      <td>0</td>
      <td>26415</td>
      <td>2016-06-24</td>
    </tr>
  </tbody>
</table>
<p>420 rows × 4 columns</p>
</div>




```python
inspections_per_day
```




    40



In order to test the effectiveness of our model in reducing the number of days it takes until establishments are caught for inspection violations we simulated the inspection process under the ranking determined by our model output and compared it with the actual inspections made in 2016. The metric used to compare our model to the actual inspection results is "average fail days". This is the average number of days it takes to inspect business that fail inspections, measured from the day they enter the universe of establishments that can possibly be inspected.

The tuned Adaboost model reduced the number of days it took to inspect establishments that failed by 44 days, at 108 vs. 152 in the actual 2016 data. However this simulation has several drawbacks which have inflated the apparent superieority of the model. The simulation assumes that on the first day of 2016 all inspections that will be made during 2016 are fully known to the model. In reality, inspections that result from complaints and food poisoning incidents are not known in advance. Thus, the universe of observable future inspections should be limited. We know from analyzing the inspections of 2016 that the city of Chicago completes on average 69 inpsections per day and so the rate of that new inspections are discovered should be on this order of magnitude, but without more knowledge of the inspection discovery process it is impossible to build a simulation whose results would be apples to apples comparison with the actual inspections from 2016. The problem can be thought of as follows: what our ranking model does is shift failed inspections forward in time within the time interval being considered. If 2016 is split into many small inspection intervals, each with their own limited information regarding future inspections, the overall forward shift of failures is small. Thus, the performance of the model depends on the size of the lookahead time interval. The problem is further complicated by the fact that different types of inspections with different failure rates have different lookahead times.

Furthermore, a true "average fail days" metric should be calculated from a start date that indicates the first day a restaurant could have failed an inspection in order to reflect the actual number of days that consumers were at risk. In order to calculate this metric we would need know when establishments qualify for inspection.

Given these considerations, the true test for our ranking model is to run it in practice, as the order in which inspections appear and are performed has an impact on which inspections are made in the future.


```python

```
