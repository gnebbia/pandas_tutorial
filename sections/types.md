
We can view and inspect types of a dataframe with:

```python
ds.dtypes
```

In order to change a column to a categoric type we can do:

```python
ds['column_name'] = ds['column_name'].astype('category', categories=['good', 'very good', 'excellent'])
```

