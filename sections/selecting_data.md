
In pandas we generally select data through the use of two methods:

* loc, which selects by using labels
* iloc, which selects by using integer numbers
* ix, which selects using an index

We may also avoid the usage of these methods and let pandas infer if
we are selecting by label or by an index integer number, but this is
not adviced, it is always better to be specific to not make the code look
ambiguous.

## Selecting with Labels (i.e., loc)

In order to select by labels we use the loc method:
```python
ds.loc[0:4, ['column1','column2']]
```

This can be considered another way to remove columns and just keep those in which
we are interested:
```python
ds.loc[:, ['column1','column2']]
```

```python
ds.loc[0:4, 'column1':'column2']
```

```python
df.loc[:, df.columns.str.startswith('foo')]
```

If we have indexes which are not integer, we can take advantage of loc
capabilities, e.g.:

```python
df.loc['2016-01-11', ['column1', 'column2']]
```

## Selecting and changing a specific value

If we want to modify the value in column 'b' which is on the first row we can do:

```python
df.loc[1, 'b'] = 'XXXXXX'
```

## Selecting with Numbers (i.e., iloc)
We can use iloc if we want to select data referring to numbers for
columns like:

```python
ds.iloc[:, 0:4]
```

We can also combine iloc and loc with:

```python
army.loc['Arizona'].iloc[2]
```
This will select the row with the index name called 'Arizona' and the
third column belonging to this raw



## Selecting with Indexes(i.e., ix) (this is deprecated)

Let's say we want to print the row with the maximum value for a specific column, we can do:
```python
max_index = df.columnname.idxmax()
df.ix[max_index]
```

