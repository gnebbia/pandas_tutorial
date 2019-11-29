Generally pandas will print only a subset of all the rows, in order to keep the
screen clean and not have a huge output, anyway sometimes, we need to really
print the entire dataframe on screen, in these cases we can simply use the
`to_string()` method accompanied by a `print()` call.

```python
print(ds.to_string())
```

