
We can compute the pearson correlation index between two columns with:
```python
Top15['column1'].corr(Top15['column2'])
```
By default pandas compute the Pearson correlation, but we can compute
other kinds of correlation indexes by specifying other options, such as:

```python
Top15['column1'].corr(method='spearman', Top15['column2'])
Top15['column1'].corr(method='kendall', Top15['column2'])
# This happens by default
Top15['column1'].corr(method='pearson', Top15['column2'])

```


We can show the correlation matrix using Pearson's Correlation Index with:
```python
import matplotlib.pyplot as plt
plt.matshow(dataframe.corr())
```



