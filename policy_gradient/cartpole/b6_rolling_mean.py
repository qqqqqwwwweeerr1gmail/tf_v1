

ls = []

for i in range(100):
    ls.append(10)

print(ls)



ls[40] = 10000


import pandas as pd

se = pd.Series(ls)
pd.set_option('display.max_rows', None)
ro = se.rolling(window=50)
# rolling_list = ro.tolist()

print(ro)

kk = se.rolling(window=50).mean()
print(kk)











