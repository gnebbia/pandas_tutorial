

These instructions can be used to describe a dataset in general:
```python
ds.columns # gets the values of columns in the dataset
ds.describe(include = "all")
ds.info()
ds.memory_user(deep = True)
```



If we want to better analyze a categorical variable, showing
a count for each category we can do:
```python
ds['col_name'].value_counts()
```

We can also show a normalized count by doing:
```python
ds['col_name'].value_counts(normalize=True)
```

Of course if we combine these methods with the `head()` method
we can get the top `n` values for a category, an example may be:
```python
ds['col_name'].value_counts(normalize=True).head(5)
# in this case we show the top five categories
```

We can also get a cumulative sum for the categories by having a percentage
value by doing:
```python
ds['col_name'].value_counts(normalize=True).cumsum().head(5)
# in this case we get the percentage of frequency for the top 5 values
# in the category called `col_name`
```

