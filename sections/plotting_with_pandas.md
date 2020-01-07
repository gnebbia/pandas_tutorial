Let's see some plotting which is generally done with pandas,
when I have to do plots I prefer to generally do:
```python
import pandas as pd
import matplotlib.pyplot as plt
```
## Line Plots

If we have a dataframe in which we can plot more columns as lines we can do:
```python
a.plot(x = 'col1', y = ['col2','col3'])
```
This will plot automatically a figure with a legend and on the x axis we will have the
values belonging to col1 while on y axis with different colors we will have the values
of col2 and col3.

If we do not specify the parameter 'x', matplotlib will automatically use the dataframe index as 'x'.

By default the plot() function uses as parameter 'kind' the value 'line', so automatically plots a line plot.

## Scatter Plots

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

## Histogram Plots

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

### Plotting the Cumulative Distribution
We can plot the cumulative distribution of a column, like this:

```python
df.column1.plot(kind='hist',
        bins=100,
        title='Cumulative distributions',
        normed=True,
        cumulative=True,
        alpha=0.4)
```


## Plotting an estimate of the Probability Density Function

In statistics, kernel density estimation (KDE) is a non-parametric way to estimate
the probability density function of a random variable. Kernel density estimation
is a fundamental data smoothing problem where inferences about the population are made,
based on a finite data sample. In some fields such as signal processing and econometrics
it is also termed the Parzen–Rosenblatt window method.


```python
df.col1.plot(kind='kde')
```

## Box Plots

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


## Bar Plots
```python
ds.column_name.plot(kind = 'bar')
```

We can also build stacked bar plot by using the parameter `stacked=True`.

Let's take an example to show how to build stacked bar plots.

Let's say we have a database of vulnerabilities with many fields, among
these fields we have `year_published` tracking the year a certain CVE
was released, and `protocol` tracking the protocol the CVE affects.

We can plot the number of CVEs by year using for each year a different
color representing a different protocol.
In order to do this we can do:
```python
stacked_data = cves.groupby(['year_published', 'protocol'])['year_published'].count().unstack('protocol').fillna(0)

# We create the handle of the figure to be able to move the legend of the plot
fig = plt.figure()

# build the stacked plot
stacked_data.plot(kind='bar', stacked=True, ax=fig.gca())

# we want to put the legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# show the plot
plt.show()
```



## Combination of more plots

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

## Scatter Matrix Plots

We can also plot scatter plots for all the features:

```python
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')
```
This not only allows us to have a lot of plots, but puts on the diagonal the probability
density function estimation with the KDE method, we can change this by putting 'hist'.

## Pie Plots

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

## Hexbin Plots

```python
df.plot(kind='hexbin', x='x', y='y', bins=100, cmap='rainbow')
```


## Correlation Plots

In order to view a correlation plot we can do:

```python
import matplotlib.pyplot as plt
plt.matshow(df.corr())
```

## Parallel Coordinates Plot
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


## Lag Plots
Lag plots are used to check if a data set or time series is random. Random data should not
exhibit any structure in the lag plot. Non-random structure implies that the underlying data are not random.

```python
lag_plot(data)
```

## Autocorrelation Plots
Autocorrelation plots are often used for checking randomness in time series.
This is done by computing autocorrelations for data values at varying time lags.

Autocorrelation plots are often used for checking randomness in time series.
This is done by computing autocorrelations for data values at varying time lags.
If time series is random, such autocorrelations should be near zero for any and all time-lag separations.
If time series is non-random then one or more of the autocorrelations will be significantly non-zero.
The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands.
The dashed line is 99% confidence band.

## Decorating Plots

We can add lines to indicate points or regions with:
```python
# draws a vertical line
plt.axvline(0.2, color='r')
# draws an horizontal line
plt.axhline(0.5, color='b')
```

## Visualizing Unstructured Data
In order to visualize unstructured data (e.g., audio, immages, text, ...), we can
make use of common packages generally used along with pandas.

### Audio

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

### Images
We can visualize images with:
```python
from PIL import Image
import numpy as np

img = Image.open('../path/name.jpg')
imgarray = np.asarray(img) # This gives us an array
imgarray.shape # with this we can understand the shape
```

At this point we could use ravel() or reshape() to change the size as we wish.


## Setting Plot Options

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

## Other Plotting Utilities

We can instantiate a new plot with a title by doing:
```python
import matplotlib.pyplot as plt

plt.figure("title of the figure")
# This states, create a plot with 3 figures, and position
# them vertically
# the general structure is subplot(nrows, ncols, index)

# here we will position the figure in the structure 3,1
# at index 1
plt.subplot(311)
# To set a scale on y axis we can use
plt.ylim([0,350])
ds0.plot()

# here we will position the figure in the structure 3,2
# at index 2
# To set a scale on y axis we can use
plt.ylim([0,350])
plt.subplot(312)
ds1.plot()

# here we will position the figure in the structure 3,3
# at index 3
# To set a scale on y axis we can use
plt.ylim([0,350])
plt.subplot(313)
ds2.plot()

```

We can also choose a stylesheet, for example we can have the same style of
the infamous ggplot package in R with:
```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
```


