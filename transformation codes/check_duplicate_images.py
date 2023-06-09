from difPy import dif

search = dif(["malignant", "benign"])
print(search.result)
print("\n")
print(search.stats)