
Pandas support different kinds of files, some good rules of thumb when reading files are:

* Specify which are the quotechars, that is, if the separator is a space, but
space can appear in some of the fields delimited by the '"' characters, then
the '"' character is our quotechar
* Specify possible escape characters, if inside some of the fields characters are escaped,
remember to specify this in the "escapechar" parameter
* Specify dtypes through the "dtype" named parameter in order to achieve some speed up
due to the fact that inferring a type takes time


## Reading a CSV File

We can read a csv file in this way:

```python
ds = pd.read_csv(filename, sep=None, engine='python', parse_dates=['fcast_date','timestamp'], dtype={'user_id': "category", 'stringa':'object'})
```

Basically we set engine to python anytime we deal with regexes.

Let's see another example:

```python
# In this case we are also setting an index column
ds = pd.read_csv("reuters_random_sample.csv", parse_dates=['time', 'published_time'],  index_col='time')
```

let's see another example:

```python
# in this case we skip the initial space we have in fields, this is very useful
# since many times we have csv files where fields are separated by a space other
# than commas to increase readability
ds = pd.read_csv("reuters_random_sample.csv", parse_dates=['time', 'published_time'],  index_col='time', skipinitialspace = True)
```

Let's see another example where we want to exclude some columns or change the
order of the existing columns:

```python
# in this case we read the cols but then switch the order in our dataframe
ds = pd.read_csv(data, usecols=['foo', 'bar'], skipinitialspace=True)[['bar', 'foo']]
```

we can also refer to columns numerically, for example:

```python
ds = pd.read_csv(data, usecols=[0,1], comment='#')
```
In this last case we also specified that lines starting with "#" have to be
considered comments, hence not to be analyzed.

Let's see another example, in this case we have fields separated by a bunch of
spaces, but still spaces can appear in some of the fields because there are
strings, for example:

1 "a string exampel" 12:32 "awdaw ddwd wa da  "
2 "a string exampel, dwao9*(0323" 12:35 "a a  awdaw ddwd, wa,, da  "

In this case we can read the file, by denoting the quoting char, so inside
quoting chars the separator can apper and will not cause any problems

```python
ds = pd.read_csv("data.csv", sep='\s+', engine='python', quotechar='"')
```

Another example could be when we have multiple separators, at this point we can
try with:

```python
# In this case we consider both ; and , as separators
df = pd.read_csv("file.csv", sep="[;,]", engine='python')
```

```python
ds = pd.read_csv("dataset.csv", engine='python', quotechar='!', header=None, names=['time','offset','title','link'], index_col='time')
```


### Reading an XLS(X) file

```python
energy = pd.read_excel("Energy Indicators.xls")
```

### Reading a Complex file

Sometimes, specifying delimiters and quotechars is not enough, we also need to
specify how characters are escaped, for example in apache web logs,
is not so uncommon to find escaped characters inside strings, for example things like:

```text
5.5.5.5 - - [03/Feb/2018:00:59:13 +0200] "GET /path/strnage\"path HTTP/1.1" 503 245520 "-" "Chrome\"Strange\"UA"
```
a string like this, can definitely confuse the parser, we should in these cases parse it like:

```python
ds = pd.read_csv("access.1.log", escapechar="\\", quotechar='"', header=None)
```

Other times, it may be still more complex, and it can be a good idea to take
advantage of regexes in order to parse a file, like this:

```python
logs = pd.DataFrame(columns=['time', 'article_id', 'user_id'])
# regc = re.compile(r'\[(?P<time>.*?)\] "GET (.*?=)(?P<article_id>\d+)(&.*?=)(?P<user_id>\d+)')
# alternative regexp that might be more efficient
regc = re.compile(r'\[(?P<time>.+)\] "GET (?:.+article_id=)(?P<article_id>\d+)(?:&user_id=)(?P<user_id>\d+)')

for line in log_file:
    m = regc.match(line)
    time = m.group('time')
    article_id = m.group('article_id')
    user_id = m.group('user_id')
    logs.append([time, article_id, user_id])
```

## Writing to a CSV File

Let's see how to save our dataframe to a new csv file:

```python
# In this case we do not want to save the index to the file
ds.to_csv(filename, index = False)
```

