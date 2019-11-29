
## Counting Duplicates
In order to count all the duplicated values we do:
```python
ds.duplicated().sum()
```

In order to count all the duplicated values with respect to a certain subset of fields we do:
```python
ds.duplicated(subset['age','zip_code']).sum()
```

In order to count duplicates with respect to a certain column we can do:
```python
ds['column_name'].duplicated().sum()
```

## Visualizing Duplicates
In order to view all the duplicates we can do:
```python
ds.loc[users.duplicated(keep = 'last'), :]
```
where keep = 'last' means that we are showing the last encountered
instance of a duplicate row.

## Enumerating Dataset Duplicates

Given the subset of duplicated rows extracted by doing:

```python
ds = ds.loc[users.duplicated(keep = 'last'), :]
```
We can count how many those are repeated with:

```python
dfanalysis = ds.groupby(['ColA','ColB']).size().reset_index(name='count')
# we can also use groupy(list(df)) if we want to use all the columns
```

we can order results in a descending way by doing:

```python
dfanalysis.sort_values(by="count", ascending=False).to_csv("duplicated_count_stats.csv", index=False)
```

## Removing Duplicates

To remove duplicates and just keep the first encountered instances we do:
```python
ds.drop_duplicates(keep = 'first')
```

To remove duplicates and just keep the last encountered instances we do:
```python
ds.drop_duplicates(keep = 'last')
```

To remove duplicates with respect to a subset of fields:
```python
ds.drop_duplicates(subset = ['age', 'zip_code'])
```

