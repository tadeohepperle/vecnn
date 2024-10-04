import pandas as pd
from squasher import Experiment

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Linux Libertine O", "sans-serif"]
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = 12

def thesis_exp(prefix: str, name: str |None = None) -> Experiment:
    if name is None:
        name = prefix
    return Experiment("./thesis_experiments/" + prefix, name = name)
def df_print(df: pd.DataFrame):
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000, "display.width",1000, "display.max_colwidth", 1000):
        print(df)
def save_plot(name: str, tight : bool = True):
    if tight:
        plt.tight_layout()
    plt.savefig("../../writing/images/" + name + ".pdf", bbox_inches='tight',pad_inches = 0.1, dpi = 200)
    plt.clf()
def model_name(params_str: str) -> str:
    for prefix in ["Ensemble", "RNNGraph", "Hnsw_faiss", "Hnsw_hnswlib", "Hnsw_rustcv", "Hnsw_jpboth", "Hnsw_vecnn", "Stitching"]:
        if params_str.startswith(prefix):
            return prefix
    return "unknown"
def model_kind(params_str: str) -> str:
    if params_str.startswith("Ensemble"):
        return "Ensemble"
    if params_str.startswith("RNNGraph"):
        return "RNNGraph"
    if params_str.startswith("Hnsw"):
        return "Hnsw"
    if params_str.startswith("Stitching"):
        return "Stitching"
    return "unknown"
# /////////////////////////////////////////////////////////////////////////////
# SECTION: read and shape the data
# /////////////////////////////////////////////////////////////////////////////


class TenMExperiments:
    a10m: Experiment
    params_and_ef: pd.DataFrame
    ef_to_recall: pd.Series
    ef_to_search_ms : pd.Series
    ef_to_search_ndc: pd.Series
    recall_gt_80: pd.DataFrame
    pivoted: pd.DataFrame

    def print(self):
        print("A10M:")
        self.a10m.print()
        print("Params and ef:")
        df_print(self.params_and_ef)
        print("Ef to recall:")
        df_print(self.ef_to_recall)
        print("Ef to search ms:")
        df_print(self.ef_to_search_ms)
        print("Ef to search ndc:")
        df_print(self.ef_to_search_ndc)
        print("Recall gt 80:")
        df_print(self.recall_gt_80)

    def __init__(self):
        a10m = thesis_exp("a10m")
        a10m.df["count"] = 1
        efs = a10m.df["ef"].unique()
        params = a10m.df["params"].unique()
        params.sort()
        
        # These things are what we are interested in:
        # 1. a frame grouped by params and ef and then flattened, with a search_time, search_ndc and search_recall columns for every params-ef combination
        # 2. 3 frames, one for recall, one for ndc and one for search_time, all grouped by params with one column for every ef
        # 2. a frame grouped by params with a build_time column and columns (recall, ndc, search_time) for the ef where recall is > 0.8
        a10m.df["model"] = a10m.df["params"].apply(model_name)
        a10m.df["model_kind"] = a10m.df["params"].apply(model_kind)
        a10m.df["threaded"] = (a10m.df["threaded"] == "True")
        
        AGGREGATE = {
            "count": "sum", 
            "search_ms": "mean", 
            "search_ndc": "mean", 
            "search_recall": "mean", 
            "build_secs": "mean",
        }
        PARAM_COLS = ["max_chunk_size", "n_vp_trees", "level_norm", "outer_loops", "m_max", "m_max0", "same_chunk_m_max", "threaded", "model", "model_kind"]
        for param in PARAM_COLS:
            AGGREGATE[param] = "first"
        
        params_and_ef = a10m.df.groupby(["params", "ef"]).agg(AGGREGATE)
        params_and_ef.reset_index(inplace=True)
        assert len(params_and_ef) == len(efs) * len(params)
        pivoted = params_and_ef.pivot(index=["params", "build_secs"] + PARAM_COLS, columns="ef")
        self.pivoted = pivoted
        ef_to_recall = pivoted["search_recall"]
        ef_to_search_ms = pivoted["search_ms"]
        ef_to_search_ndc = pivoted["search_ndc"]
        
        recall_gt_80 = pivoted[("count", 30)].to_frame()
        recall_gt_80.reset_index(inplace=True)
        recall_gt_80.drop(columns=[("count", 30)], inplace=True)
        recall_gt_80["params"] = recall_gt_80["params"].astype(str)
        
        recall_gt_80["ef"] = 0
        recall_gt_80["recall"] = 0
        recall_gt_80["search_ms"] = 0
        recall_gt_80["search_ndc"] = 0
        
        MIN_RECALL = 0.8
        for i, row in recall_gt_80.iterrows():
            params_str = row['params'].iloc[0] # completely retarded, but this is how you access the string stored in the column 
            # find the ef where recall is > 0.8
            searches = params_and_ef[params_and_ef["params"] == params_str]
            last_ef = 0
            last_recall = 0
            last_search_ms = 0
            last_search_ndc = 0
            last_count = 0
            for j, s in searches.iterrows():
                assert s["ef"] > last_ef
                last_ef = s["ef"]
                last_count = s["count"]
                last_search_ms = s["search_ms"]
                last_search_ndc = s["search_ndc"]
                last_recall = s["search_recall"]
                if s["search_recall"] > MIN_RECALL:
                    break
            recall_gt_80.at[i, "count"] = last_count   
            recall_gt_80.at[i, "ef"] = last_ef
            recall_gt_80.at[i, "recall"] = last_recall
            recall_gt_80.at[i, "search_ms"] = last_search_ms
            recall_gt_80.at[i, "search_ndc"] = last_search_ndc
        self.a10m = a10m
        self.ef_to_recall = ef_to_recall
        self.ef_to_search_ms = ef_to_search_ms
        self.ef_to_search_ndc = ef_to_search_ndc
        self.recall_gt_80 = recall_gt_80
        self.params_and_ef = params_and_ef


    def recall_lower_80(self) -> pd.DataFrame:
        return self.recall_gt_80[exp.recall_gt_80["recall"] <= 0.8][["ef","recall","search_ms", "build_secs", "params"]]
    def recall_greater_80(self) -> pd.DataFrame:
        return self.recall_gt_80[exp.recall_gt_80["recall"] > 0.8][["ef","recall","search_ms", "build_secs", "params", "threaded", "count"]]
    def recall_greater_80_single_threaded(self) -> pd.DataFrame:
        return self.recall_gt_80[(exp.recall_gt_80["recall"] > 0.8) & (exp.recall_gt_80["threaded"] == False)][["ef","recall","search_ms", "build_secs", "params"]]
    def recall_greater_80_multi_threaded(self) -> pd.DataFrame:
        return self.recall_gt_80[(exp.recall_gt_80["recall"] > 0.8) & (exp.recall_gt_80["threaded"] == True)][["ef","recall","search_ms", "build_secs", "params"]]
    def recall_greater_80_compare(self) -> pd.DataFrame:
        objs = {}
        gt = self.recall_greater_80()
        for params_str in gt["params"].unique():
            row = gt[(gt["params"] == params_str)]
            params_str = str(params_str)
            threaded = False
            if "threaded: True" in params_str:
                threaded = True
                params_str = params_str.replace(", threaded: True", "")
            else:
                params_str = params_str.replace(", threaded: False", "")
            
            if params_str not in objs:
                objs[params_str] = {"params": params_str, "both": 0}
            objs[params_str]["both"] +=1
            if threaded:
                objs[params_str]["threaded_recall"] = row["recall"].iloc[0]
                objs[params_str]["threaded_build_secs"] = row["build_secs"].iloc[0]
                objs[params_str]["threaded_search_ms"] = row["search_ms"].iloc[0]
                objs[params_str]["threaded_ef"] = row["ef"].iloc[0]
            else:
                objs[params_str]["single_recall"] = row["recall"].iloc[0]
                objs[params_str]["single_build_secs"] = row["build_secs"].iloc[0]
                objs[params_str]["single_search_ms"] = row["search_ms"].iloc[0]   
                objs[params_str]["single_ef"] = row["ef"].iloc[0]     
        objs_list = [v for k,v in objs.items()]
        return pd.DataFrame(objs_list, columns=["params", "both", "threaded_build_secs", "threaded_search_ms", "threaded_recall", "threaded_ef","single_build_secs", "single_search_ms", "single_recall", "single_ef"])


# /////////////////////////////////////////////////////////////////////////////
# SECTION: Display the results
# /////////////////////////////////////////////////////////////////////////////
exp= TenMExperiments()

# remove all the experiments that do not cross 80 recall
not_crossing_80 = exp.recall_gt_80[exp.recall_gt_80["recall"] < 0.8]["params"]
print("These params are too bad, never cross 80 recall:")
for p in not_crossing_80:
    print("    ", p)
crossing_80 = exp.recall_gt_80[exp.recall_gt_80["recall"] > 0.8]["params"]
print("These params are good enough reaching more than 80 recall:")
for p in crossing_80:
    print("    ", p)


print("\n\nRecall lower than 80 (at ef=150):")
df_print(exp.recall_lower_80())

print("\n\nRecall greater than 80:")
df_print(exp.recall_greater_80())

print("\n\nRecall greater than 80, multithreaded:")
df_print(exp.recall_greater_80_multi_threaded())

print("\n\nRecall greater than 80, singlethreaded:")
df_print(exp.recall_greater_80_single_threaded())

print("\n\nRecall greater than 80, compare single and multi threaded:")
df_compare = exp.recall_greater_80_compare()
df_print(df_compare)
print("\n\nRecall greater than 80, compare single and multi threaded where both have been executed:")
df_print(df_compare[df_compare["both"] > 1])

MODEL_KIND_COLOR = {
    "Ensemble": "green",
    "RNNGraph": "blue",
    "Hnsw": "red",
    "Stitching": "orange",
}

MODEL_KIND_NAME = {
    "Ensemble": "VP-Tree Ensemble",
    "RNNGraph": "Relative NN-Descent",
    "Hnsw": "HNSW",
    "Stitching": "Stitching",
}


print("\n\nFor all runs that have been performed multiple times, validate that the differences are not large:")
df = exp.a10m.df
show_ef = 150
df = df[df["ef"] == show_ef][["params", "ef", "search_recall", "search_ms", "build_secs"]]
vc = df['params'].value_counts()
vc = vc[vc > 1]
vc = vc.index
df = df[df["params"].isin(vc)]
df.sort_values(by="params",inplace=True)
df_print(df)


print("\n\nRecall Increase for each param combination over ef 30..150:")
df_print(exp.ef_to_recall)

# df = df.pivot(index=["params"], columns="ef")
# df_print(df)
# df_print(exp.pivoted[exp.pivoted[("count", 30)] > 1])

params_and_ef_filtered = exp.params_and_ef[exp.params_and_ef["params"].isin(crossing_80) & (exp.params_and_ef["search_ms"] < 6)]
# print(params_and_ef_filtered)
# params_and_ef_filtered = exp.params_and_ef


# fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.5))
# axs[0].grid(axis="y")
# for params in params_and_ef_filtered["params"].unique():
#     subdf = params_and_ef_filtered[params_and_ef_filtered["params"] == params]
#     kind: str = str(subdf["model_kind"].iloc[0])
#     axs[0].plot(subdf["ef"], subdf["search_recall"], color=MODEL_KIND_COLOR[kind], label=kind, alpha=0.5)
#     axs[0].scatter(subdf["ef"], subdf["search_recall"], label=kind, color=MODEL_KIND_COLOR[kind], alpha=0.5)
# axs[0].set_xlabel("ef")
# axs[0].set_ylabel("Recall")
# axs[0].set_xticks([30,60,90,120,150])

# for params in params_and_ef_filtered["params"].unique():
#     subdf = params_and_ef_filtered[params_and_ef_filtered["params"] == params]
#     kind: str = str(subdf["model_kind"].iloc[0])
#     axs[1].plot(subdf["ef"], subdf["search_ms"], color=MODEL_KIND_COLOR[kind], label=kind, alpha=0.5)
#     axs[1].scatter(subdf["ef"], subdf["search_ms"], label=kind, color=MODEL_KIND_COLOR[kind], alpha=0.5)
# axs[1].set_xlabel("ef")
# axs[1].set_xticks([30,60,90,120,150])
# axs[1].set_ylabel("Search time (ms)")

# handles, labels = axs[0].get_legend_handles_labels()
# labels = [MODEL_KIND_NAME[l] for l in labels]
# by_label = dict(zip(labels, handles))
# fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
# plt.subplots_adjust(top=0.8)
# save_plot("a10m_recall_and_search_ms_by_kind", tight = False)


# for kind in ["Ensemble", "RNNGraph", "Hnsw"]:
#     subdf = exp.recall_gt_80[(exp.recall_gt_80["model_kind"] == kind) & (exp.recall_gt_80["threaded"] == True)]
#     plt.scatter(subdf["build_secs"], subdf["search_ms"], label=kind, color=MODEL_KIND_COLOR[kind])
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
# plt.show()

# df80 = exp.recall_gt_80.copy()
# fig, axs = plt.subplots(1, 2, figsize=(16.8, 8.5))
# axs[0].grid(axis="y")
# for par in exp.recall_gt_80["params"].unique():
#     subdf = df80[df80["params"] == par]
#     axs[0].plot(subdf["build_secs"], subdf["search_ms"], label=subdf["model"].iloc[0])
#     axs[0].scatter(subdf["build_secs"], subdf["search_ms"])
# axs[0].set_xlabel("build_secs")
# axs[0].set_ylabel("search_ms")

# # axs[1].grid(axis="y")
# # for par in df80["params"].unique():
# #     subdf = df80[df80["params"] == par]
# #     axs[1].plot(subdf["ef"], subdf["search_ms"], label=par)
# #     axs[1].scatter(subdf["ef"], subdf["search_ms"])
# # axs[1].set_xlabel("ef")
# # axs[1].set_ylabel("Search time (ms)")

# # add a legend on top of the 2 plots with the key colors (remove the duplicates)
# handles, labels = axs[0].get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
# # add more padding to top of figure such that legend does not overlap plots:
# plt.subplots_adjust(top=0.8)
# plt.show()
