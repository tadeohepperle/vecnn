
import pandas as pd

df = pd.read_csv("out.txt")

df = df[10:]

print(df)

import matplotlib.pyplot as plt

plt.plot(df["d"], df["(max-min)/min"])
plt.show()