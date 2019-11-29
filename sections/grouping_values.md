Let's say we have a dataset which tells us various characteristics for neighbourhoods
for different cities, let's say now we want to understand things related to cities,
a way to do this is to use groupby.
By using groupby we can imagine, that we are splitting our dataframe in multiple smaller
dataframes, each related to a specific thing.
For example taking our city and neighbourhoods dataframe, a groupby on the column 'city',
would produce different dataframes, each one containing only rows related to a specific city.

## Printing a groupby dataframe
```python
g = df.groupby('city')

for city, city_df in g:
    print(city)
    print(city_df)

```
An alternative way could be:

```python
g.get_group("Napoli")
```

With the last command meaning "print only the sub dataframe related to the city Napoli".

A groupby basically implements a sequence of the following operations:

* split, splits the original DataFrame in sub DataFrames
* apply, applies an operation, we can ask for max(), min(), mean() and whatever other function
* combine, again in a new Series or DataFrame


```python
ds.groupby('column_name').column2.mean()
```

```python
ds.groupby('column_name').column2.max()
```

```python
ds.groupby('continent').mean()
```

```python
ds.groupby('city').describe()
```

We can also apply these statistics for each row, for example let's say
that we have a dataset and many columns represent years, while the values are
some energy spent through these years, we can compute the average or max, min by using
the axis = 1 parameter, e.g.:
```python
years = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
ds['avg_energy'] = Top15[years].mean(axis = 1)
```


We can also plot data for each different city for example:
```python
ds.groupbby('city').plot()
```

We can also use multiple columns in the groupby for example, let's say that
we want to print the mean age for each combination of occupation and gender,
we can do:

```python
users.groupby(['occupation','gender']).mean().age
```


