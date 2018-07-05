# Pandas Tutorial

In pandas we have two datastructure:
    *Series (which can be thought as an array)
    *DataFrame (which can be thought as a matrix)

A DataFrame can be also viewed as an array of Series,
dataframes and series are flexible, and can contain labels for fields,
indexes and many other interesting features which would be complicated
to have on plain arrays/matrices.

## Creating a Dataframe
```python
df1 = pd.DataFrame({
    "city": ["new york","chicago","orlando"],
    "temperature": [21,14,35],
     })
```

## Describe a dataset 
```python 
$1.describe(include = "all")
$1.info()
$1.memory_user(deep = True)
```

## Types
We can view and inspect types of a dataframe with:
```python 
ds.dtypes
```

In order to change a column to a categoric type we can do:
```python 
ds['column_name'] = ds['column_name'].astype('category', categories=['good', 'very good', 'excellent'])
```

## Basic Pandas Statistics

```python 
ds.field_name.mean()
```

```python 
ds.field_name.median()
```

```python 
ds.field.quantile([0.1,0.15, .9])
```

## Manipulating CSV Files

### Reading a CSV File
```python 
ds = pd.read_csv($2, sep=None, engine='python', parse_dates=['fcast_date','timestamp'], dtype={'user_id': "category", 'stringa':'object'})
```

### Writing to a CSV File
```python 
ds.to_csv($2, index = False)
```

## Selecting Data

### Selecting with Labels

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

### Selecting and changing a specific value
If we want to modify the value in column 'b' which is on the first row we can do:
```python
df.loc[1, 'b'] = 'XXXXXX'
 ```

### Selecting with Numbers
We can use iloc if we want to select data referring to numbers for
columns like:

```python 
ds.iloc[:, 0:4]
```

## Filters
```python 
ds[(ds.column1 >= 200) & (ds.column2 == 'Drama')]
```

## Pandas Conditionals

```python 
df.loc[df.AAA >= 5,['BBB','CCC']] = 555;
```

```python pd_if_else
df['logic'] = np.where(df['AAA'] > 5,'high','low'); df
```

## Column Operations

### Remove columns
```python 
ds.drop(['column1','column2'], 1, inplace = True)
```

### Remove Column on a Condition
```python 
c = c[c.n_opts != 5]
```

### Rename columns
```python
ds.rename(columns={'fcast_date_a': 'date_fcast'}, inplace=True)
```

### Create new Columns
```python 
ds["days_from_start"] = ds["fcast_date_a"] - ds["date_start"]
```

### Create new Columns with Apply
```python 
def compute_euclidean_distance(row):
	a = np.array([row['value_a'], row['value_b'], row['value_c']])
	b = np.array([row['a'], row['b'], row['c']])
	return distance.euclidean(a, b)

ds['new_distance'] = ds.apply(compute_euclidean_distance, axis=1)
```

### Create Dummy Columns for One-Hot Encoding
```python 
one_hot_cols = pd.get_dummies(ds['outcome'], prefix='outcome')
ds.drop('outcome', axis=1, inplace = True)
ds = ds.join(one_hot_cols)
```

### Create a Dataframe as a combination of two dataframes with different columns
 The main purpose of a cross-tabulation is to enable readers to readily compare two categorical variables.

```python pd_rowconcat
ds = pd.concat([df_even, df_odd], axis=1)
```

## Row Operations

```python pd_unique
ds["ifp_id"].unique()
```
### Split a Dataset into Train/Test 

```python pd_traintestsplit
train = dataset.sample(frac=0.95,random_state=200)
test = dataset.drop(train.index)
```

### Concatenate rows of two different datasets with same columns
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


## Merge

In order to merge on a field which could be considered a primary key we can do:
```python
c = pd.merge(ds1, ds2, on='ifp_id')
```
Now this is by default an inner join, that means, that only the 'ifp\_id' which
are intersection of both ds1 and ds2 are taken into account.

We can do an outer join by specifying the attribute called 'how'.

```python
df3 = pd.merge(df1,df2,on="city",how="outer")
```

We can also specify if we want to keep all the keys containes only in the left dataset
or right dataset with:
```python
df3 = pd.merge(df1,df2,on="city",how="left")
```

If we have column names which are shared by both datasets we can easily add suffixes,
for example:
```python
df3 = pd.merge(df1,df2,on="city",how="outer", suffixes=('_first','_second'))
```
or let's say we have a couple of predictions, so user in table 1 has value1 value2 value3
and also a user in table 2, so we can do:
```python
df3 = pd.merge(df1,df2,on="ifp_id",how="inner", suffixes=('_user1','_user2'))
```


## Dealing with Null Values

### Summarizing Null Values
```python 
ds.isna().sum()
```

### Removing Null Values
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

### Reaplacing Null Values

```python 
ds['column_name'].fillna(value='not assigned', inplace = True)
```


```python 
ds['columnname'].value_counts(dropna = False)
```

## Dealing with Duplicates

### Counting Duplicates
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

### Visualizing Duplicates
In order to view all the duplicates we can do:
```python 
ds.loc[users.duplicated(keep = 'last'), :]
```
where keep = 'last' means that we are showing the last encountered 
instance of a duplicate row

### Removing Duplicates

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

## Sorting Values 

```python 
ds.sort_values(['column_1'], ascending=False)
```

## Grouping Values
```python 
ds.groupby('column_name').column2.mean()
```

```python 
ds.groupby('column_name').column2.max()
```

```python 
ds.groupby('continent').mean()
```

## Map, Apply and ApplyMap
Map applies a translation to each element of a series:

```python 
ds['new_column'] = ds.column.name.map({'female':0, 'male':1})
```

Apply applies a function to each element of a series
```python 
ds['new_column'] = train.col1.apply(len)
```

## Cross Tab
The main purpose of a cross-tabulation is to enable readers to readily compare two categorical variables:

```python
pd.crosstab(ds.column_x, ds.column_y)
```

## Plotting with Pandas
```python
ds.column_name.plot(kind = 'hist')
```

```python
ds.column_name.plot(kind = 'bar')
```

