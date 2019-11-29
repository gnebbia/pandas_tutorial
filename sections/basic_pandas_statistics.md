
```python
ds.field_name.mean()
```

```python
ds.field_name.median()
```

```python
ds.field.quantile([0.1,0.15, .9])
```

We can plot more percentiles with the commodity of list comprehensions,
the following will plot all the percentiles:
```python
quantiles_lst =  [x * 0.01 for x in range(0, 101)]
ds.field.quantile(quantiles_lst)
```

The following will print the deciles like:
0.1, 0.2, 0.3, ...
```python
quantiles_lst =  [x * 0.1 for x in range(0, 11)]
ds.field.quantile(quantiles_lst)
```

The following will print the percentiles multiples of 5 like:
0.05, 0.1, 0.15, 0.2, ...

```python
quantiles_lst =  [x * 0.01 for x in range(0, 101) if x % 5 == 0]
ds.field.quantile(quantiles_lst)
```


