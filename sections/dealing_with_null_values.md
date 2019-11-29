
## Summarizing Null Values
```python
ds.isna().sum()
```

## Removing Null Values
In order to drop all the rows which have a null value on *any* field we do:
```python
ds.dropna(how='any')
```

In order to drop all the rows which have a null value on *all* the field we do:
```python
ds.dropna(how='all')
```

In order to drop all the rows which have a null value on *any* field within a subset we do:
```python
ds.dropna(subset = ['column', 'column2'], how='any')
```

In order to drop all the rows which have a null value on *all* the fields within a subset we do:
```python
ds.dropna(subset = ['column', 'column2'], how='all')
```

## Reaplacing Null Values

```python
ds['column_name'].fillna(value='not assigned', inplace = True)
```


```python
ds['columnname'].value_counts(dropna = False)
```

