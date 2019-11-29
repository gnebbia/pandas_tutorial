
A time series is a set of data points indexed in time oder, for example stock prices during a year,
or a specific physical value in time.

In order to parse date correctly we can specify our own customm function to deal with dates:
```python
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
```
Let's say for example that our dates in a file are surrounded by square brackets as in apache web server
logs, at this point we could also strip those characters.

We can also defer the parsing and setting of a time/date field by doing:
```python
data['time'] = pd.to_datetime(data['time'], format = "%Y%m%d %I:%M %p")
# sometimes the format can be auto inferred by pandas
# data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)
```

## Time Series Aggregation

Let's say we want to aggregate our time series by hour, or by minute, or by day,
we can do it using the `resample` method.

Let's say we just have a bunch of timestamps, which do not have a specific
structure, like by minute, or by hour and so on.
For example they could represent the access times made to a web page,
so we basically have timestamps without any other information.

Like this:

```
2007-05-05 18:51:37
2007-05-05 18:54:02
2007-05-05 19:59:11
2007-05-05 19:59:11
2007-05-05 19:59:11
2007-05-06 22:33:18
2007-10-26 08:17:42
```

We can transform this data to a timeseries with a specific resolution in this
way:

```python
# we first set a 1 to each timestamp, which is useful for the aggregation
# into a time series
ds['count'] = 1

# these are some examples of possible aggregations
ds_minute = ds.resample('T').sum() # minute
ds_15minute = ds.resample('15T').sum() # 15 minutes
ds_hour = ds.resample('H').sum() # hour
ds_day = ds.resample('D').sum() # day
ds_week = ds.resample('W').sum() # week
ds_month = ds.resample('M').sum() # month
ds_year = ds.resample('A').sum() # year
```

## Time Series Common Tasks: Converting Date Format


### From Unix Time to Human Readable Date

```python
df['date'] = pd.to_datetime(df['date'],unit='s')
```

In order to convert to Unix Time a Human Readable date, we can do:

```python
ds['time'] = (ds['time'].astype(np.int64)/1e9).astype(np.int64)
```



## Time Series Common Tasks: Getting the Day of the Week

Sometimes it can be useful to get the weekdays to be able to divide our dataset
into working week days and weekend days. This can be easily achieved with:

```python
series['day_of_week'] = series.index.weekday_name
ds_week = series[~series['day_of_week'].isin(['Saturday','Sunday'])]
ds_weekend = series[series['day_of_week'].isin(['Saturday','Sunday'])]
# Now we can remove the fields of the name if we don't need them
del series['day_of_week']
del ds_week['day_of_week']
del ds_weekend['day_of_week']
```

## Time Series Common Tasks: Filtering a time series with dates

We can filter a time seris in this way:
```python
date_mask = (ds.index >= "2010-05-01") & (ds.index < "2010-07-01")
ds[date_mask]
```

Now we are saying take all the days starting from the first of may,
(this is included) until the last day of june.

I think there is no difference in terms of dates between > and >=.

We can filter data by dates like in multiple ways, let's see another example:
```python
# We pick all the data points from the beginning of 2015 to the end of 2016
date_mask = (ds_utc.index >= "2015-01-01") & (ds_utc.index < "2017-01-01")

# Now we take from 2012 to 2014
ds_utc_2y = ds_utc[date_mask]
```

Notice that if our time series has the timestamp/date as index we can also use
the function "slice" to perform a filter, such as:

```sh
period = slice('2017-07-17 00:00:00','2018-07-16 23:59:00')
ds.loc[period]
```


## Time Series Common Tasks: Converting time in different units

If we want to have the difference in hours between to pandas datetimes we can
do:

```python
ds['difference_in_hours'] = (ds['published_time'] - ds.index).astype('timedelta64[h]')
```

If we have a timedelta and just want to convert it into an integer number of seconds,
we can do:

```python
df['duration_seconds'] = df['duration'] / np.timedelta64(1, 's')
```


