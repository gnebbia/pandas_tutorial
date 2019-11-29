
## Getting the maximum among more columns

To create an additional column which is the maximum among different columns we
can simply do:
```python
dss['top_topic_value'] = dss[['topic_0','topic_1','topic_2']].max(axis=1)
```

Anyway what if we need to get the column name which has the maximum value?
In this case we can simply use the `idxmax` method, like this:

```python
dss['top_topic'] = dss[['topic_0','topic_1','topic_2']].idxmax(axis=1)
```


## Cumulative Sum of a Column

Given a column, we can build a cumulative sum of the column by using:

```python
ds['cum_sum'] = ds.columnname.cumsum()
ds['cum_perc'] = 100*ds.cum_sum/ds.columnname.sum()

ds.cum_perc.plot() # plots the cumulative distribution
```
