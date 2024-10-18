import pandas as pd
from squasher import Experiment
from results_utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
plt.rcParams["font.family"] = ["Linux Libertine O", "sans-serif"]
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('axes', axisbelow=True)
plt.rcParams["font.size"] = 12

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

MODEL_IDENTIFIERS = [
    ("Ensemble6", "Ensemble {level_norm: 0.0, n_vp_trees: 6, n_candidates: 0, max_chunk_size: 256, same_chunk_m_max: 10, m_max: 20, m_max0: 40}"),
    ("Ensemble6Layered", "Ensemble {level_norm: 0.3, n_vp_trees: 6, n_candidates: 0, max_chunk_size: 256, same_chunk_m_max: 10, m_max: 20, m_max0: 40}"),
    ("Ensemble10", "Ensemble {level_norm: 0.0, n_vp_trees: 10, n_candidates: 0, max_chunk_size: 256, same_chunk_m_max: 10, m_max: 20, m_max0: 40}"),
    ("Ensemble6Rnn", "Ensemble {level_norm: 0.0, n_vp_trees: 6, n_candidates: 0, max_chunk_size: 1024, same_chunk_m_max: 10, m_max: 20, m_max0: 40, rnn_inner_loops: 3, rnn_outer_loops: 2}"),
    ("Ensemble10Rnn", "Ensemble {level_norm: 0.0, n_vp_trees: 10, n_candidates: 0, max_chunk_size: 1024, same_chunk_m_max: 10, m_max: 20, m_max0: 40, rnn_inner_loops: 3, rnn_outer_loops: 2}"),
    ("RnnDescent2x3", "RNNGraph {outer_loops: 2, inner_loops: 3, m_pruned: 40, m_initial: 40}"),
    ("RnnDescent3x3", "RNNGraph {outer_loops: 3, inner_loops: 3, m_pruned: 40, m_initial: 40}"),
    ("Hnsw30", "Hnsw_vecnn {ef_constr: 30, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("Hnswlib30", "Hnsw_hnswlib {ef_constr: 30, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("JpBoth30", "Hnsw_jpboth {ef_constr: 30, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("RustCv30", "Hnsw_rustcv {ef_constr: 30, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("Faiss30", "Hnsw_faiss {ef_constr: 30, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("Hnsw40", "Hnsw_vecnn {ef_constr: 40, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("Hnswlib40", "Hnsw_hnswlib {ef_constr: 40, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("JpBoth40", "Hnsw_jpboth {ef_constr: 40, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("RustCv40", "Hnsw_rustcv {ef_constr: 40, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("Faiss40", "Hnsw_faiss {ef_constr: 40, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("Hnsw60", "Hnsw_vecnn {ef_constr: 60, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("Hnswlib60", "Hnsw_hnswlib {ef_constr: 60, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("JpBoth60", "Hnsw_jpboth {ef_constr: 60, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("RustCv60", "Hnsw_rustcv {ef_constr: 60, m_max: 20, m_max0: 40, level_norm: 0.3}"),
    ("Faiss60", "Hnsw_faiss {ef_constr: 60, m_max: 20, m_max0: 40, level_norm: 0.3}"),
]
IDENTIFIER_TO_MODEL_KIND = {i: model_kind(p) for i, p in MODEL_IDENTIFIERS}
MODEL_IDENTIFIER_NAMES = [i for i, p in MODEL_IDENTIFIERS]
PARAMS_TO_IDENTIFIER = {p: i for i, p in MODEL_IDENTIFIERS}

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
# /////////////////////////////////////////////////////////////////////////////
# SECTION: read and shape the data
# /////////////////////////////////////////////////////////////////////////////

COMPARE_SAME_EF = 100 # recall reached at ef 60, for other comparision mode
RECALL_EF_COL = f"recall_ef{COMPARE_SAME_EF}"
SEARCH_MS_EF_COL = f"search_ms_ef{COMPARE_SAME_EF}"
MIN_RECALL = 0.8
class TenMExperiments:
    a10m: Experiment
    params_and_ef: pd.DataFrame
    ef_to_recall: pd.Series
    ef_to_search_ms : pd.Series
    ef_to_search_ndc: pd.Series
    recall_agg: pd.DataFrame
    recall_agg_by_threaded: pd.DataFrame
    pivoted: pd.DataFrame
    id_models: pd.DataFrame

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
        print("recall_agg:")
        df_print(self.recall_agg)

    def __init__(self):
        a10m = thesis_exp("a10m_filtered")
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
        
        recall_agg = pivoted[("count", 30)].to_frame()
        recall_agg.reset_index(inplace=True)
        recall_agg.drop(columns=[("count", 30)], inplace=True)
        recall_agg["params"] = recall_agg["params"].astype(str)
        
        recall_agg["ef"] = 0
        recall_agg["recall"] = 0
        recall_agg["search_ms"] = 0
        recall_agg["search_ndc"] = 0
        recall_agg[RECALL_EF_COL] = 0
        recall_agg[SEARCH_MS_EF_COL] = 0
        
        for i, row in recall_agg.iterrows():
            params_str = row['params'].iloc[0] # completely retarded, but this is how you access the string stored in the column 
            # find the ef where recall is > 0.8
            searches = params_and_ef[params_and_ef["params"] == params_str]
            last_ef = 0
            last_recall = 0
            last_search_ms = 0
            last_search_ndc = 0
            last_count = 0
            min_recall_reached : bool = False
            for j, s in searches.iterrows():
                assert s["ef"] > last_ef
                if s["ef"] == COMPARE_SAME_EF:
                    recall_agg.at[i, RECALL_EF_COL] = s["search_recall"]
                    recall_agg.at[i, SEARCH_MS_EF_COL] = s["search_ms"]
                if not min_recall_reached:
                    last_ef = s["ef"]
                    last_count = s["count"]
                    last_search_ms = s["search_ms"]
                    last_recall = s["search_recall"]
                    # last_search_ndc = s["search_ndc"]
                if s["search_recall"] > MIN_RECALL:
                    min_recall_reached = True
                
            recall_agg.at[i, "count"] = last_count   
            recall_agg.at[i, "ef"] = last_ef
            recall_agg.at[i, "recall"] = last_recall
            recall_agg.at[i, "search_ms"] = last_search_ms
            # recall_agg.at[i, "search_ndc"] = last_search_ndc
            
        self.a10m = a10m
        self.ef_to_recall = ef_to_recall
        self.ef_to_search_ms = ef_to_search_ms
        self.ef_to_search_ndc = ef_to_search_ndc
        self.recall_agg = recall_agg
        self.params_and_ef = params_and_ef

        objs = {}
        for params_str in recall_agg["params"].unique():
            row = recall_agg[(recall_agg["params"] == params_str)]
            params_str = str(params_str)
            threaded = False
            if "threaded: True" in params_str:
                threaded = True
                params_str = params_str.replace(", threaded: True", "")
            else:
                params_str = params_str.replace(", threaded: False", "")            
            if params_str not in objs:
                objs[params_str] = {"params": params_str}
            prefix = "threaded" if threaded else "single"
            objs[params_str][prefix + "_recall"] = row["recall"].iloc[0]
            objs[params_str][prefix + "_cnt"] = row["count"].iloc[0]
            objs[params_str][prefix + "_build_secs"] = row["build_secs"].iloc[0]
            objs[params_str][prefix + "_search_ms"] = row["search_ms"].iloc[0]
            objs[params_str][prefix + "_ef"] = row["ef"].iloc[0]
            objs[params_str][prefix + "_" + RECALL_EF_COL] = row[RECALL_EF_COL].iloc[0]
            objs[params_str][prefix + "_" + SEARCH_MS_EF_COL] = row[SEARCH_MS_EF_COL].iloc[0]

        objs_list = [v for k,v in objs.items()]
        agg_by_treaded_colums = ["params", "both", "threaded_build_secs", "threaded_search_ms", "threaded_recall", "threaded_ef", "threaded_cnt", "single_build_secs", "single_search_ms", "single_recall", "single_ef",  "single_cnt", f"threaded_{RECALL_EF_COL}", f"threaded_{SEARCH_MS_EF_COL}", f"single_{RECALL_EF_COL}", f"single_{SEARCH_MS_EF_COL}",]
        self.recall_agg_by_threaded = pd.DataFrame(objs_list, columns=agg_by_treaded_colums)
        self.recall_agg_by_threaded["threaded_cnt"] = self.recall_agg_by_threaded["threaded_cnt"].fillna(0).astype(int)
        self.recall_agg_by_threaded["single_cnt"] = self.recall_agg_by_threaded["single_cnt"].fillna(0).astype(int)
        self.recall_agg_by_threaded["both"] = (self.recall_agg_by_threaded["threaded_cnt"] >= 1) & (self.recall_agg_by_threaded["single_cnt"] >= 1)
        self.recall_agg_by_threaded["identifier"] = self.recall_agg_by_threaded["params"].map(PARAMS_TO_IDENTIFIER)

        self.id_models  = self.recall_agg_by_threaded[~self.recall_agg_by_threaded["identifier"].isna()]
        self.id_models .drop(columns=["params"], inplace=True)
        self.id_models .insert(0, 'identifier', self.id_models .pop('identifier') ) 
        self.id_models .sort_values(by="identifier",inplace=True)

    def recall_lower_80(self) -> pd.DataFrame:
        return self.recall_agg[exp.recall_agg["recall"] <= 0.8][["ef","recall","search_ms", "build_secs", "params"]]
    def recall_greater_80(self) -> pd.DataFrame:
        return self.recall_agg[exp.recall_agg["recall"] > 0.8][["ef","recall","search_ms", "build_secs", "params", "threaded", "count"]]

def convert_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    hours = str(hours)+"h" if hours > 0 else ""
    minutes = str(minutes)+"m" if minutes > 0 else ""
    secs = str(secs)+"s" if secs > 0 else ""
    if secs == "" and (minutes != "" or hours != ""):
        secs = "0s"
    return hours, minutes, secs


def id_models_latex_df(exp: TenMExperiments, single: bool) -> pd.DataFrame:
    st = "single" if single else "threaded"
    df = exp.id_models[["identifier", f"{st}_build_secs", f"{st}_ef", f"{st}_recall", f"{st}_search_ms", f"{st}_{RECALL_EF_COL}", f"{st}_{SEARCH_MS_EF_COL}"]]
    df.rename(columns={"identifier": "identifier", f"{st}_build_secs": "build_time", f"{st}_ef": "ef", f"{st}_recall": "recall", f"{st}_search_ms": "search_ms", f"{st}_{RECALL_EF_COL}": RECALL_EF_COL, f"{st}_{SEARCH_MS_EF_COL}": SEARCH_MS_EF_COL}, inplace=True)
    df['identifier'] = pd.Categorical(df['identifier'], categories=MODEL_IDENTIFIER_NAMES, ordered=True)
    df = df[~df["build_time"].isna()]
    df.sort_values(by="identifier",inplace=True) 
    df[["h","m","s"]] = df["build_time"].apply(convert_seconds).apply(pd.Series)
    
    df["recall"] = df["recall"].round(3)
    df[RECALL_EF_COL] = df[RECALL_EF_COL].round(3)
    df["search_ms"] = df["search_ms"].round(3)
    df["search_rank"] = df["search_ms"].rank(ascending=True).astype(int)
    # add a rank column based on build time
    df["rank"] = df["build_time"].rank(ascending=True).astype(int)
    df["ef"] = df["ef"].astype(int)
    return df
def latex_table_for_id_models_single_threaded(exp: TenMExperiments):
    df = id_models_latex_df(exp, single=True)
    df = df[["identifier", "h", "m", "s", "rank", "ef", "recall", "search_ms", "search_rank"]]
    df["search_ms"] = df["search_ms"].round(3).apply(lambda x: f"{x:.3f}ms")
    df["identifier"] = df["identifier"].apply(lambda x: f"\emph{{{x}}}")
    df.to_latex("./tables/a10m_id_models_single.tex", index=False, escape=False, header=False, float_format="%.3f")
    df_print(df)

def latex_table_for_id_models_multi_threaded(exp: TenMExperiments):
    df = id_models_latex_df(exp, single=False)
    df_single = id_models_latex_df(exp, single=True)
    df["search_ms"] = df["search_ms"].round(3).apply(lambda x: f"{x:.3f}ms")
    df["identifier"] = df["identifier"].apply(lambda x: f"\emph{{{x}}}")
    df["speedup"] = (df_single["build_time"] / df["build_time"]).apply(lambda x: "-" if pd.isna(x) else f"{x:.1f}x")
    df = df[["identifier", "m", "s", "speedup","rank",  "ef", "recall", "search_ms", "search_rank"]]
    df.to_latex("./tables/a10m_id_models_threaded.tex", index=False, escape=False, header=False, float_format="%.3f")
    df_print(df)


def latex_table_for_id_models_multi_threaded_fixed_ef(exp: TenMExperiments):
    df = id_models_latex_df(exp, single=False).copy()
    df_single = id_models_latex_df(exp, single=True)
    df[SEARCH_MS_EF_COL] = df[SEARCH_MS_EF_COL].round(3).apply(lambda x: f"{x:.3f}ms")
    df["identifier"] = df["identifier"].apply(lambda x: f"\emph{{{x}}}")
    # df["speedup"] = (df_single["build_time"] / df["build_time"]).apply(lambda x: "-" if pd.isna(x) else f"{x:.1f}x")
    df["search_ms_rank"] = df[SEARCH_MS_EF_COL].rank(ascending=True).astype(int)
    df["recall_rank"] = df[RECALL_EF_COL].rank(ascending=False).astype(int)
    df = df[["identifier", "m", "s", "rank", RECALL_EF_COL, "recall_rank", SEARCH_MS_EF_COL, "search_ms_rank"]]
    df["rank_sum"] = df["recall_rank"] + df["search_ms_rank"] + df["rank"]
    df["rank_sum"] = (df["rank_sum"] - df["rank_sum"].min()) / (df["rank_sum"].max() - df["rank_sum"].min())
    df["rank_sum"] = (1.0 -df["rank_sum"]).round(2).apply(lambda x: f"{x:.2f}")
    
    df.to_latex("./tables/a10m_id_models_threaded_by_fixed_ef.tex", index=False, escape=False, header=False, float_format="%.3f")
    df_print(df)

def df_data_to_latex(df: pd.DataFrame, file_name: str):
    s = ""
    for i, row in df.iterrows():
        for i, col in enumerate(df.columns):
            if i == len(df.columns) - 1:
                s += str(row[col]) + "\\\\ \n"
            else:
                s += str(row[col]) + " & "
    with open(file_name, "w") as f:
        f.write(s)

# /////////////////////////////////////////////////////////////////////////////
# SECTION: Display the results
# /////////////////////////////////////////////////////////////////////////////
exp= TenMExperiments()

not_crossing_80 = exp.recall_agg[exp.recall_agg["recall"] < 0.8]["params"]
print("These params are too bad, never cross 80 recall:")
for p in not_crossing_80:
    print("    ", p)
crossing_80 = exp.recall_agg[exp.recall_agg["recall"] > 0.8]["params"]
print("These params are good enough reaching more than 80 recall:")
for p in crossing_80:
    print("    ", p)

print("\n\nRecall lower than 80 (at ef=150):")
df_print(exp.recall_lower_80())

print("\n\nRecall greater than 80:")
df_print(exp.recall_greater_80())

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


print("\n\nRecall aggregated by threaded:")
df_print(exp.recall_agg_by_threaded)

print("\n\nRecall aggregated by threaded for the FINAL models:")
df_print(exp.id_models)

latex_table_for_id_models_single_threaded(exp)
latex_table_for_id_models_multi_threaded(exp)
latex_table_for_id_models_multi_threaded_fixed_ef(exp)

df = id_models_latex_df(exp, single=True)
df["model_kind"] = df["identifier"].map(IDENTIFIER_TO_MODEL_KIND)


dfm = id_models_latex_df(exp, single=False)
dfm["model_kind"] = dfm["identifier"].map(IDENTIFIER_TO_MODEL_KIND)





fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.5), sharey=True)  # Share y-axis
set_page_fract(2)
axs[0].barh(df["identifier"], df["build_time"]/60, label=df["model_kind"], color=df["model_kind"].map(MODEL_KIND_COLOR))
axs[0].invert_yaxis()
axs[0].invert_xaxis()
axs[0].yaxis.set_label_position("right") 
axs[0].yaxis.tick_right()               
axs[0].set_xlabel("Single-threaded build time (m)")
axs[1].barh(dfm["identifier"], dfm["build_time"]/60, label=dfm["model_kind"], color=dfm["model_kind"].map(MODEL_KIND_COLOR))
axs[1].invert_yaxis()
axs[1].set_xlabel("Multi-threaded build time (m)")
fig.tight_layout(pad=0.4)
fig.savefig("../../writing/images/a10m_build_time.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300) #


dfm_no_faiss = dfm.copy()
dfm_no_faiss = dfm_no_faiss[(dfm_no_faiss["search_ms"] < 8) & (dfm_no_faiss["recall"] >= 0.8)]
print("dfm:")
df_print(dfm)
df_print(dfm_no_faiss)

set_page_fract(1)
# plt.rcParams["font.size"] = 10
# plt.scatter(dfm_no_faiss["build_time"]/60, dfm_no_faiss["search_ms_ef100"], c=dfm_no_faiss["recall_ef100"])
plt.scatter(dfm_no_faiss["build_time"]/60, dfm_no_faiss["search_ms"], color=dfm_no_faiss["model_kind"].map(MODEL_KIND_COLOR))
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xticks(range(0, 15, 1))
plt.xlabel("Build time (minutes)")
plt.ylabel("Search time (ms)")


for row in dfm_no_faiss.itertuples():
    build_mins = row.build_time/60
    search_ms = row.search_ms
    x_pos = build_mins
    y_pos = search_ms
    if row.model_kind == "Ensemble":
        if "Layered" in row.identifier:
            x_pos -= 0.4
            y_pos += 0.2
        elif row.identifier == "Ensemble6":
            x_pos += 0.8
            y_pos += 0.0
        else:
            x_pos += 0.3
            y_pos += 0.1
    if row.model_kind == "RNNGraph":
        x_pos -= 3.0
        y_pos -= 0.2
    if row.model_kind == "Hnsw":
        x_pos += 0.3
        y_pos += 0.1
    if row.identifier == "Hnsw30":
        y_pos += 0.05
    plt.annotate(row.identifier, (build_mins, search_ms), xytext=(x_pos, y_pos), arrowprops=dict(arrowstyle="->", connectionstyle="angle3"), fontsize=11 )

plt.savefig("../../writing/images/a10m_multi_threaded_pareto.pdf", bbox_inches='tight',pad_inches = 0.1, dpi = 300)
# plt.show()

# plt.barh(df["identifier"], df["recall_ef100"], label = df["model_kind"], color=df["model_kind"].map(MODEL_KIND_COLOR))
# plt.gca().invert_yaxis()
# plt.xlabel("Recall at ef=100")
# plt.show()

# print("\n\nRecall Increase for each param combination over ef 30..150:")
# df_print(exp.ef_to_recall)

# df = df.pivot(index=["params"], columns="ef")
# df_print(df)
# df_print(exp.pivoted[exp.pivoted[("count", 30)] > 1])

params_and_ef_filtered = exp.params_and_ef[exp.params_and_ef["params"].isin(crossing_80) & (exp.params_and_ef["search_ms"] < 6)]
fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.5))
axs[0].grid(axis="y")
unique_params = params_and_ef_filtered["params"].unique()
# random.shuffle(unique_params)
for params in unique_params:
    subdf = params_and_ef_filtered[params_and_ef_filtered["params"] == params]
    kind: str = str(subdf["model_kind"].iloc[0])
    axs[0].plot(subdf["ef"], subdf["search_recall"], color=MODEL_KIND_COLOR[kind], label=kind, alpha=0.4)
    axs[0].scatter(subdf["ef"], subdf["search_recall"], label=kind, color=MODEL_KIND_COLOR[kind], alpha=0.4)
axs[0].set_xlabel("ef")
axs[0].set_ylabel("Recall")
axs[0].set_xticks([30,60,90,120,150])
for params in unique_params:
    subdf = params_and_ef_filtered[params_and_ef_filtered["params"] == params]
    kind: str = str(subdf["model_kind"].iloc[0])
    axs[1].plot(subdf["ef"], subdf["search_ms"], color=MODEL_KIND_COLOR[kind], label=kind, alpha=0.4)
    axs[1].scatter(subdf["ef"], subdf["search_ms"], label=kind, color=MODEL_KIND_COLOR[kind], alpha=0.4)
axs[1].set_xlabel("ef")
axs[1].set_xticks([30,60,90,120,150])
axs[1].set_ylabel("Search time (ms)")
handles, labels = axs[0].get_legend_handles_labels()
labels = [MODEL_KIND_NAME[l] for l in labels]
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
plt.subplots_adjust(top=0.8)
save_plot("a10m_recall_and_search_ms_by_kind", tight = False)


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
