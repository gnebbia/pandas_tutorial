
We can combine dataset by using merge, as in database theory we must understand
what it means to do:

*   Outer Join (Union), we take everything, this is the equivalent of a union
*   Inner Join (Intersection), we take only things which are in both sets
    (i.e., dataframes) this is the equivalent of an intersection
*   Conditional Joins (Left and Right Joins), this is the equivalent of an
    intersection with a union with one of the sets, or dataframes

By default pandas performs inner joins.


## Outer Join

```python
pd.merge(df1, df2, how = 'outer', left_index = True, right_index = True)
```

## Inner Join

```python
pd.merge(df1, df2, how = 'inner', left_index = True, right_index = True)
```


In order to merge on a field which could be considered a primary key we can do:
```python
c = pd.merge(ds1, ds2, on='ifp_id')
```

or

```python
print(pd.merge(products, invoices, left_index=True, right_on='Product ID'))
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

## Removing Rows
We can remove rows, for example in order to remove header and footer information
from a dataset we can do:

```python
ds = ds[8:246]
```
this will take from the 8th row to the 245th row of the dataset.

## Conditional Joins (Left and Right)
Let's say that df1 is a dataset related to the staff of a university
while df2 is the dataframe related to the students.

We can create a new dataframe containing all the staff and information about
students only if the staff members are students with a left join:

```python
pd.merge(df1, df2, how = 'left', left_index = True, right_index = True)
```

In the other case, if we want to have all students but include information
for the ones who are of the staff (who is not belonging to the staff will have
these info at Null we can do:

```python
pd.merge(df1, df2, how = 'right', left_index = True, right_index = True)
```

