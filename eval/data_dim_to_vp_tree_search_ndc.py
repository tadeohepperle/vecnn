import vecnn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Linux Libertine O", "sans-serif"]
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (4,3) 

n = 1000
n_q = 100
n_trees = 10

rec = []
for dims in range(2,21):
    data = np.random.random((n,dims)).astype("float32")
    queries = np.random.random((n_q,dims)).astype("float32")
    ds = vecnn.Dataset(data)
    ndc= 0
    for seed in range(n_trees):
        vp_tree = vecnn.VpTree(ds, seed)
        for i in range(n_q):
            ndc += vp_tree.knn(queries[i,:],1).num_distance_calculations
    ndc /= n_q
    ndc /= n_trees
    rec.append({"dims": dims, "ndc": ndc, "percent": ndc / n })

df = pd.DataFrame(rec)

# generate a line that shows dims on the x axis and percent on the y axis, there should be ticks for all whole numbers on the x axis
plt.plot(df["dims"], df["percent"])
plt.xlabel("Dimensions")
plt.ylabel("Fraction of data visited")
plt.xticks(range(2,21,3))
plt.tight_layout()
plt.savefig("results/data_dim_to_vp_tree_search_ndc.pdf")
df.to_csv("results/data_dim_to_vp_tree_search_ndc.csv")
