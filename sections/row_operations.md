
```python
ds["ifp_id"].unique()
```

## Deleteing Rows which have missing values

```python
df.dropna()
```

## Split a Dataset into Train/Test


```python
train = dataset.sample(frac=0.95,random_state=200)
test = dataset.drop(train.index)
```

## Select Rows based on Condition

```python
ds[ds['colname1'] == 'value']

# Let's select all the rows which have as value
# in colnam2 the string America or Europe
ds[ds['colnam2'].isin(['America','Europe'])]

# Now we perform a negation of the previous filter with
# the character '~'
ds[~ds['colnam2'].isin(['America','Europe'])]
```


## Concatenate rows of two different datasets with same columns
In order to concatenate rows of more datasets we can basically do:
```python
pd.concat([df1, df2, df3], ignore_index = True)
```

A useful shortcut to concat() are the append() instance methods on Series and DataFrame.
These methods actually predated concat. They concatenate along axis=0, namely the index:
```python
result = df1.append(df2, ignore_index = True)
```

Another example of this, is when our dataset is split among more files,
in this case we can do:
```python
frames = [ process_your_file(f) for f in files ]
result = pd.concat(frames, ignore_index = True)
```


