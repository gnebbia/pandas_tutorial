
We can check if two series or dataframe are equal, i.e., they have the same
values with:

```python
assert ds['columnname'].equals(ds2['anothercolumn'])
```

Another example using dataframes instead of series may be:

```python
ds.equals(ds2)
```


