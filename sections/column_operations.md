
## Remove columns

```python
ds.drop(['column1','column2'], 1, inplace = True)
```
or easier with:
```python
del crime['column1']
```

## Remove Column on a Condition

```python
c = c[c.n_opts != 5]
```

## Rename columns
```python
ds.rename(columns={'fcast_date_a': 'date_fcast'}, inplace=True)
```

## Moving Columns

Sometimes we want to change the ordering of columns, this can be useful
especially in visualization or manual inspection contexts, in order to change
column order, we can simply grab the column names as a list and then apply
whatever operation on the list and put it back on our dataframe.
Let's see an example, where we want to put the column called "session_id" as our
first column, we can do:

```python
cols = list(ds) # we can also grab the column names list with ds.columns.tolist()
# here we insert the "session_id" column in the zeroth position, that is
# the first element
cols.insert(0, cols.pop(cols.index('session_id')))

# now we overwrite our dataframe
ds = ds[cols]
```


## Create new Columns

```python
ds["days_from_start"] = ds["fcast_date_a"] - ds["date_start"]
```

## Create new Columns with Apply

```python
def compute_euclidean_distance(row):
	a = np.array([row['value_a'], row['value_b'], row['value_c']])
	b = np.array([row['a'], row['b'], row['c']])
	return distance.euclidean(a, b)

ds['new_distance'] = ds.apply(compute_euclidean_distance, axis=1)
```

## Create new Columns based on difference of Rows

We can use the shift function to create a new dataset which is shifted
by one position, for example, in the case where our dataset represents
HTTP requests arriving at a webserver, we can compute the interarrival
column by just doing:

```python
ds['diff'] = ds['time_in_sec'] - ds['time_in_sec'].shift(1)
```

of course the first element will be a NaN, which we have to deal with,
since it has no corresponding element to perform the subtraction in this case.

Another common usage of the shift function is when we want to create a dataset
which can be used with an AR model or in general with time series.

This can be done for example like this:

```python
def create_sequence_ds(ds, colname_to_shift, num_steps_backward):
        for i in range(num_steps_backward):
            ds['shift_'+str(i)] = ds[colname_to_shift].shift(i+1)
        return ds
```


## Inspect Column Values

How many items for each category in a column?
```python
df.column_name.value_counts()
```
How many different items for a specific category in a column?
```python
df.column_name.value_counts().count()
```
or faster with:
```python
df.column_name.nunique()
```

Remember that *value_counts* is useful for ordering a categorical variable
while *sort_values* is useful when ordering a numerical variable or a categorical
variable for which an order is specified.

Let's see a couple of examples:
```python
df.column_name_cat.value_counts(ascending = False)
```
while for a numerical variable we can do:
```python
df.column_name_cat.sort_values(ascending = False)
```


## Create Dummy Columns for One-Hot Encoding

```python
one_hot_cols = pd.get_dummies(ds['outcome'], prefix='outcome')
ds.drop('outcome', axis=1, inplace = True)
ds = ds.join(one_hot_cols)
```

## Create Dummy Columns for Dummy Encoding


```python
one_hot_cols = pd.get_dummies(ds['outcome'], prefix='outcome', drop_first=True)
ds.drop('outcome', axis=1, inplace = True)
ds = ds.join(one_hot_cols)
```

## Create a categorical variable from a continuous variable

We can create ranges for continuous variable to transform them into categorical
variables, in pandas we can do this with:
```python
ds['RenewCat'] = pd.cut(ds['% Renewable'], bins=5)
```
In this case we are using 5 bins, of course we can use more bins and have more categories.

If we precisely know the interval values to which we want to perform the split we can do:

```python
ds['newcol'] = pd.cut(ds['age'], bins=[0, 18, 35, 70])
```
notice that the intervals are inclusive, so the first one will go from 0 to 18 included, while
the second one will go from 18 excluded to 35 included and so on.

Other ways to discretize features are using numpy with:

```python
discretized_age = np.digitize(age, bins=[20,30,64])
```

In this last case if we have a series called age which is made like this:
`6, 12, 20, 36, 65`, after the operation, digitized_age will be like this:
`0, 0, 1, 1, 1`.

So the bin numbers are exclusive.


## Create a Dataframe as a combination of two dataframes with different columns

 The main purpose of a cross-tabulation is to enable readers to readily compare two categorical variables.

```python
ds = pd.concat([df_even, df_odd], axis=1)
```

