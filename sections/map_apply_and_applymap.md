Map, Apply and ApplyMap operations are used to perform transformations
on a dataset.

Map applies a translation to each element of a series:

```python
ds['new_column'] = ds.column.name.map({'female':0, 'male':1})
```
an alternative to map is to insert all the substitutions in a dictionary and then use
replace, like this:

```python
subs = {"Republic of Korea": "South Korea",
"United States of America": "United States",
"United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
"China, Hong Kong Special Administrative Region": "Hong Kong"}

ds.replace({'column_name":subs}, inplace = True)
```

Apply applies a function to each element of a series
```python
ds['new_column'] = train.col1.apply(len)
```
let's see another example:
```python
energy['Country'] = energy['Country'].apply(remove_digit)
```

We use *apply* everytime we want to build new columns based for example
on values of the rows, for example, let's say we want to build a new column
which keeps track of the average of the fields dist1, dist2 and dist3
we can do:

```python
def avg_dist(row):
    row['avg_dist'] = (row['dist1'] + row['dist2'] + row['dist3']) / 3
    return row

df.apply(avg_dist, axis = 1)
```

Let's see another example where we add a column called *legal_drinker* which
says if a person in a dataset can drink or not in the US:

```python
def majority(row):
    if row > 17:
        return True
    else:
        return False

df['legal_drinker'] = df.age.apply(majority)
```
an alternative way to implement it is:

```python
def majority(row):
    if row['age'] > 17:
        row['legal_drinker'] == True
    else:
        row['legal_drinker'] == False
    return row

df.apply(majority, axis =1)
```

so keep in mind that if we consider the fields of the rows specifically, we have
to add the field *axis = 1* to apply.




Another example, could be if we want to find the maximum and minimum among a set
of columns, in this case we can do:

```python
def min_max(row):
    data = [['POPESTIMATE2010',
            'POPESTIMATE2011',
            'POPESTIMATE2012',
            'POPESTIMATE2013',
            'POPESTIMATE2014',
            'POPESTIMATE2015']]
    row['max'] = np.max(data)
    row['min'] = np.min(data)

    return row

df.apply(min_max, axis = 1)
```

Note: The most commonly used method is map, while applymap is more rarely used.

Let's see another example where we use a lambda expression, or anonymous
function:

```python
# Here we change an integer column to a timedelta in hours.
ds['time_offset_col'] = ds.time_offset_col.apply(lambda x: pd.Timedelta(hours=x))
```

Let's see another example where we want to pass to the function called in apply
more arguments, this can be done using lambdas like this:

```python
def apply_labeling(x, time, other_time):
    if (x['intertime_s'] >= time and x['intertime_s2'] >= other_time):
        return 300
    else:
        return 100

ds['new_label'] = ds.apply(lambda x : apply_labeling(x, 300, 500), axis=1)
```

## Operations to perform on Groups

On dataframes where we used groupby we can generally perform different
operations, let's see some examples:

* if we want to get a single value for each group - use `aggregate()`
* if we want to get a subset of the input rows - use `filter()`
* if we want to get a new value for each input row - use `transform()`



