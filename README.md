# Pandas Tutorial

In pandas we have two datastructure:
    *Series (which can be thought as an array or as pandas 1D data structure)
    *DataFrame (which can be thought as a matrix or as pandas 2D data structure)

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

## Describe a Dataset 
```python 
ds.describe(include = "all")
ds.info()
ds.memory_user(deep = True)
```

## Print a Dataset

Generally pandas will print only a subset of all the rows, in order to keep the
screen clean and not have a huge output, anyway sometimes, we need to really
print the entire dataframe on screen, in these cases we can simply use the
`to_string()` method accompanied by a `print()` call.

```python
print(ds.to_string())
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

We can read a csv file in this way:

```python
ds = pd.read_csv(filename, sep=None, engine='python', parse_dates=['fcast_date','timestamp'], dtype={'user_id': "category", 'stringa':'object'})
```

Let's see another example:

```python
# In this case we are also setting an index column
ds = pd.read_csv("reuters_random_sample.csv", parse_dates=['time', 'published_time'],  index_col='time')
```

#### Reading an XLS(X) file

```python
energy = pd.read_excel("Energy Indicators.xls")
```

### Writing to a CSV File

Let's see how to save our dataframe to a new csv file:

```python
# In this case we do not want to save the index to the file
ds.to_csv(filename, index = False)
```

## Selecting Data

In pandas we generally select data through the use of two methods:

* loc, which selects by using labels
* iloc, which selects by using integer numbers
* ix, which selects using an index

We may also avoid the usage of these methods and let pandas infer if
we are selecting by label or by an index integer number, but this is
not adviced, it is always better to be specific to not make the code look
ambiguous.

### Selecting with Labels (i.e., loc)

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

### Selecting and changing a specific value
If we want to modify the value in column 'b' which is on the first row we can do:
```python
df.loc[1, 'b'] = 'XXXXXX'
 ```

### Selecting with Numbers (i.e., iloc)
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



### Selecting with Indexes(i.e., ix) (this is deprecated)

Let's say we want to print the row with the maximum value for a specific column, we can do:
```python
max_index = df.columnname.idxmax()
df.ix[max_index]
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
or easier with:
```python
del crime['column1']
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

### Create new Columns based on difference of Rows

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


### Inspect Column Values

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


### Create Dummy Columns for One-Hot Encoding
```python 
one_hot_cols = pd.get_dummies(ds['outcome'], prefix='outcome')
ds.drop('outcome', axis=1, inplace = True)
ds = ds.join(one_hot_cols)
```

### Create a categorical variable from a continuous variable

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


### Create a Dataframe as a combination of two dataframes with different columns
 The main purpose of a cross-tabulation is to enable readers to readily compare two categorical variables.

```python 
ds = pd.concat([df_even, df_odd], axis=1)
```

## Row Operations

```python 
ds["ifp_id"].unique()
```

### Deleteing Rows which have missing values

```python
df.dropna()
```

### Split a Dataset into Train/Test 


```python
train = dataset.sample(frac=0.95,random_state=200)
test = dataset.drop(train.index)
```

### Select Rows based on Condition

```python
ds[ds['colname1'] == 'value']]

# Let's select all the rows which have as value
# in colnam2 the string America or Europe
ds[ds['colnam2'].isin(['America','Europe'])]

# Now we perform a negation of the previous filter with 
# the character '~'
ds[~ds['colnam2'].isin(['America','Europe'])]
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

We can combine dataset by using merge, as in database theory we must understand
what it means to do:

*   Outer Join (Union), we take everything, this is the equivalent of a union
*   Inner Join (Intersection), we take only things which are in both sets
    (i.e., dataframes) this is the equivalent of an intersection
*   Conditional Joins (Left and Right Joins), this is the equivalent of an
    intersection with a union with one of the sets, or dataframes

By default pandas performs inner joins.


### Outer Join

```python
pd.merge(df1, df2, how = 'outer', left_index = True, right_index = True)
```

### Inner Join

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

### Removing Rows
We can remove rows, for example in order to remove header and footer information 
from a dataset we can do:

```python
$1 = $1[8:246]
```
this will take from the 8th row to the 245th row of the dataset.

### Conditional Joins (Left and Right)
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

## Removing Data from Datasets

In order to remove all rows where the field 'Quantity' is equal to 0 we can do:

```python
df.drop(df[df['Quantity'] == 0].index)
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

We can sort by a specific column by doing:
```python 
ds.sort_values(['column_1'], ascending=False)
```

We can also sort using multiple columns by doing:
```python
dfworking = dfworking.sort_values(['STATE','DISTRICT','GENERAL VOTE'], ascending=[True, True, False])
```

We can also take the top values for a specific dataframe with *nlargest*:
```python
df.nlargest(3, 'column_name')
```

Or the lowest values with:

```python
df.nsmallest(3, 'column_name')
```

Another way to sort a dataset by the value of a column inplace is:
```python
df.sort('rank',inplace=True)
```

## Grouping Values
Let's say we have a dataset which tells us various characteristics for neighbourhoods
for different cities, let's say now we want to understand things related to cities,
a way to do this is to use groupby.
By using groupby we can imagine, that we are splitting our dataframe in multiple smaller
dataframes, each related to a specific thing.
For example taking our city and neighbourhoods dataframe, a groupby on the column 'city',
would produce different dataframes, each one containing only rows related to a specific city.

### Printing a groupby dataframe
```python
g = df.groupby('city')

for city, city_df in g:
    print(city)
    print(city_df)

```
An alternative way could be:

```python
g.get_group("Napoli")
```

With the last command meaning "print only the sub dataframe related to the city Napoli".

A groupby basically implements a sequence of the following operations:

* split, splits the original DataFrame in sub DataFrames
* apply, applies an operation, we can ask for max(), min(), mean() and whatever other function
* combine, again in a new Series or DataFrame


```python 
ds.groupby('column_name').column2.mean()
```

```python 
ds.groupby('column_name').column2.max()
```

```python 
ds.groupby('continent').mean()
```

```python
ds.groupby('city').describe()
```

We can also apply these statistics for each row, for example let's say 
that we have a dataset and many columns represent years, while the values are
some energy spent through these years, we can compute the average or max, min by using
the axis = 1 parameter, e.g.:
```python
years = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
ds['avg_energy'] = Top15[years].mean(axis = 1)
```


We can also plot data for each different city for example:
```python
ds.groupbby('city').plot()
```

We can also use multiple columns in the groupby for example, let's say that
we want to print the mean age for each combination of occupation and gender,
we can do:

```python
users.groupby(['occupation','gender']).mean().age
```


## Map, Apply and ApplyMap
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

## Cross Tab
The main purpose of a cross-tabulation is to enable readers to readily compare two categorical variables:

```python
pd.crosstab(ds.column_x, ds.column_y)
```

## Plotting with Pandas
Let's see some plotting which is generally done with pandas,
when I have to do plots I prefer to generally do:
```python
import pandas as pd
import matplotlib.pyplot as plt
```
### Line Plots

If we have a dataframe in which we can plot more columns as lines we can do:
```python
a.plot(x = 'col1', y = ['col2','col3'])
```
This will plot automatically a figure with a legend and on the x axis we will have the
values belonging to col1 while on y axis with different colors we will have the values
of col2 and col3.

If we do not specify the parameter 'x', matplotlib will automatically use the dataframe index as 'x'.

By default the plot() function uses as parameter 'kind' the value 'line', so automatically plots a line plot.

### Scatter Plots

We can make a scatter plot of two columns of a dataframe like this:
```python
df.plot(kind='scatter', x='Height', y='Weight')
```

Now let's say we want to plot more things on the same plot, what we can do is use the 
parameter 'ax' to refer to the same plot.

For example:
```python
fig, ax = plt.subplots()
males.plot(kind='scatter', x='Height', y='Weight',
           ax=ax, color='blue', alpha=0.3,
           title='Male & Female Populations')
females.plot(kind='scatter', x='Height', y='Weight',
             ax=ax, color='red', alpha=0.3)
```
Or another thing we can do is to add to our dataframe a color column and then add the 'c' parameter:

```python
df['Gendercolor'] = df['Gender'].map({'Male': 'blue', 'Female': 'red'})
df.plot(kind='scatter', 
        x='Height',
        y='Weight',
        c=df['Gendercolor'],
        alpha=0.3,
        title='Male & Female Populations')
```

We can also specify the value range on the axis with the parameters 'xlim' and 'ylim', like this:
```python
df.plot(kind='scatter', x='col1', y='col2',
            xlim=(-1.5, 1.5), ylim=(0, 3))
```

### Histogram Plots

We can plot histograms like this:
```python
df['Height'].plot(kind='hist',
                     bins=50,
                     alpha=0.3,
                     color='blue')
```

we can also specify a range by doing:
```python
df['Height'].plot(kind='hist',
                     bins=50,
                     alpha=0.3,
                     range = (30,100),
                     color='blue')

```

We can also have the mean or median line overimposed on an histogram plot,
for example by doing:

```python
plt.axvline(males['Height'].mean(), color='blue', linewidth=2)
plt.axvline(females['Height'].mean(), color='red', linewidth=2)
```

#### Plotting the Cumulative Distribution
We can plot the cumulative distribution of a column, like this:

```
df.column1.plot(kind='hist',
        bins=100,
        title='Cumulative distributions',
        normed=True,
        cumulative=True,
        alpha=0.4)
```

### Plotting an estimate of the Probability Density Function

In statistics, kernel density estimation (KDE) is a non-parametric way to estimate 
the probability density function of a random variable. Kernel density estimation 
is a fundamental data smoothing problem where inferences about the population are made, 
based on a finite data sample. In some fields such as signal processing and econometrics 
it is also termed the Parzen–Rosenblatt window method.


```python
df.col1.plot(kind='kde')
```

### Box Plots

```python
df.column1.plot(kind='box',
        color = 'red',
        title='Boxplot')
```


We can also plot boxplots horizontally like this:
```python
df.plot.box(vert=False, positions=[1, 4, 5, 6, 8])
# here we also specified the positions
```

```python
color = dict(boxes='DarkGreen', whiskers='DarkOrange',
            medians='DarkBlue', caps='Gray')
df.plot.box(color=color, sym='r+')
```


### Bar Plots
```python
ds.column_name.plot(kind = 'bar')
```

### Combination of more plots

```python
fig, ax = plt.subplots(2, 2, figsize=(5, 5))

df.plot(ax=ax[0][0],
        title='Line plot')

df.plot(ax=ax[0][1],
        style='o',
        title='Scatter plot')

df.plot(ax=ax[1][0],
        kind='hist',
        bins=50,
        title='Histogram')

df.plot(ax=ax[1][1],
        kind='box',
        title='Boxplot')

plt.tight_layout()  # this is used in order to not have titles imposed on plots
```

### Scatter Matrix Plots

We can also plot scatter plots for all the features:

```python
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')
```
This not only allows us to have a lot of plots, but puts on the diagonal the probability
density function estimation with the KDE method, we can change this by putting 'hist'.

### Pie Plots

```python
gt01 = df['data1'] > 0.1
piecounts = gt01.value_counts()
# Piecounts will have only two values with a specific count
piecounts.plot(kind='pie',
               figsize=(5, 5),
               explode=[0, 0.15],
               labels=['<= 0.1', '> 0.1'],
               autopct='%1.1f%%',
               shadow=True,
               startangle=90,
               fontsize=16)
```

### Hexbin Plots

```python
df.plot(kind='hexbin', x='x', y='y', bins=100, cmap='rainbow')
```


### Correlation Plots

In order to view a correlation plot we can do:

```python
import matplotlib.pyplot as plt
plt.matshow(df.corr())
```

### Parallel Coordinates Plot
Parallel coordinates is a plotting technique for plotting multivariate data, 
see the Wikipedia entry for an introduction. 
Using parallel coordinates points are represented as connected line segments.
Each vertical line represents one attribute. One set of connected line segments 
represents one data point. Points that tend to cluster will appear closer together.

```python
from pandas.plotting import parallel_coordinates
plt.figure()
parallel_coordinates(df, 'Title')
```

The PCA and LDA plots are useful for finding obvious cluster in the data,
in the other side scatter plot matrices or parallel coordinate plots show specific 
behavior of features in a dataset.


### Lag Plots
Lag plots are used to check if a data set or time series is random. Random data should not 
exhibit any structure in the lag plot. Non-random structure implies that the underlying data are not random.

```python
lag_plot(data)
```

### Autocorrelation Plots
Autocorrelation plots are often used for checking randomness in time series.
This is done by computing autocorrelations for data values at varying time lags.

Autocorrelation plots are often used for checking randomness in time series. 
This is done by computing autocorrelations for data values at varying time lags. 
If time series is random, such autocorrelations should be near zero for any and all time-lag separations. 
If time series is non-random then one or more of the autocorrelations will be significantly non-zero. 
The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands. 
The dashed line is 99% confidence band.

### Visualizing Unstructured Data
In order to visualize unstructured data (e.g., audio, immages, text, ...), we can 
make use of common packages generally used along with pandas.

#### Audio

For the audio, we can see the signal with:

```python
from scipy.io import wavfile

rate, snd = wavfile.read(filename = 'nameoffile.wav')
plt.plot(snd)
```

We can also view the spectrogram by doing:
```python
_ = plt.specgram(snd, NFFT=1024, Fs=44100)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
```

#### Images
We can visualize images with:
```python
from PIL import Image
import numpy as np

img = Image.open('../path/name.jpg')
imgarray = np.asarray(img) # This gives us an array
imgarray.shape # with this we can understand the shape
```

At this point we could use ravel() or reshape() to change the size as we wish.


### Setting Plot Options

Once we have a plot with pandas:

```python
hist_plot = ds.colnam1.plot(kind='hist', bins=50)
hist_plot.set_xlim(-200,200)
hist_plot.set_xlim(-350,350)
```

Another parameter used when plotting is the label,
notice that labels support latex, so we can do:
```python
ax.plot(x, i * x, label='$y = %ix$'.format(i))
```

Or
```python
bar_plot = ds.colnam1.plot(kind='hist', bins=50)
bar_plot.set_xlabel("x label")
bar_plot.set_ylabel("y label")
```


## Correlation with Pandas

We can compute the pearson correlation index between two columns with:
```python
Top15['column1'].corr(Top15['column2'])
```


We can show the correlation matrix using Pearson's Correlation Index with:
```python
import matplotlib.pyplot as plt
plt.matshow(dataframe.corr())
```



## Time Series Analysis with Pandas

A time series is a set of data points indexed in time oder, for example stock prices during a year,
or a specific physical value in time.

In order to parse date correctly we can specify our own customm function to deal with dates:
```python
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
```
Let's say for example that our dates in a file are surrounded by square brackets as in apache web server
logs, at this point we could also strip those characters.

We can also defer the parsing and setting of a time/date field by doing:
```python
data['time'] = pd.to_datetime(data['time'], format = "%Y%m%d %I:%M %p")
# sometimes the format can be auto inferred by pandas
# data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)
```

### Time Series Common Tasks: Getting the Day of the Week

Sometimes it can be useful to get the weekdays to be able to divide our dataset
into working week days and weekend days. This can be easily achieved with:

```python
series['day_of_week'] = series.index.weekday_name
ds_week = series[~series['day_of_week'].isin(['Saturday','Sunday'])]
ds_weekend = series[series['day_of_week'].isin(['Saturday','Sunday'])]
# Now we can remove the fields of the name if we don't need them
del series['day_of_week']
del ds_week['day_of_week']
del ds_weekend['day_of_week']
```

### Time Series Common Tasks: Converting time in different units

If we want to have the difference in hours between to pandas datetimes we can
do:

```python
ds['difference_in_hours'] = (ds['published_time'] - ds.index).astype('timedelta64[h]')
```


### Filter by Date

We can filter data by dates like in multiple ways, one way is:
```python
# We pick all the data points from the beginning of 2015 to the end of 2016
date_mask = (ds_utc.index >= "2015-01-01") & (ds_utc.index < "2017-01-01")

# Now we take from 2012 to 2014
ds_utc_2y = ds_utc[date_mask]
```



## Appendix A: Pandas Options

Change the maximum number of printable rows:
```python
pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)
```



