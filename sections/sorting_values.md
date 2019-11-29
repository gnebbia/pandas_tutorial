
We can sort by a specific column by doing:
```python
ds.sort_values(['column_1'], ascending=False)
```

We can also sort using multiple columns by doing:
```python
dfworking = dfworking.sort_values(['STATE','DISTRICT','GENERAL VOTE'], ascending=[True, True, False])
```

We can also take the top values for a specific dataframe with *nlargest*:
```python
df.nlargest(3, 'column_name')
```

Or the lowest values with:

```python
df.nsmallest(3, 'column_name')
```

Another way to sort a dataset by the value of a column inplace is:
```python
df.sort('rank',inplace=True)
```

