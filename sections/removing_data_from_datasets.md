
In order to remove all rows where the field 'Quantity' is equal to 0 we can do:

```python
df.drop(df[df['Quantity'] == 0].index)
```


