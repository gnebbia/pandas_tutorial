
Change the maximum number of printable rows:
```python
pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)
```

When printing to file, we generally do not want to be limited in the number of
columns, so we should set the following option:

```python
pd.set_option('display.max_colwidth', 0)
```

We can reset any option with reset_option(), for example:

```python
pd.reset_option('display.max_rows')
```



