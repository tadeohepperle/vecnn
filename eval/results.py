"""
Generates graphs and tables from the experiment results that can be directly using in the latex thesis document.
"""
from typing import Literal
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from squasher import Experiment, save_all_experiments_as_latex_tables
from results_utils import *

set_page_fract(2)

hnsw_ef_search = thesis_exp("exp_hnsw_effect_of_ef_search", "hnsw_ef_search").filter("k", 30)
hnsw_ef_search.print()
plt.grid(axis="y")
plt.plot(hnsw_ef_search.df["ef"], hnsw_ef_search.df["recall"])
plt.scatter(hnsw_ef_search.df["ef"], hnsw_ef_search.df["recall"])
plt.yticks([i/100 for i in range(85,101,5)])
plt.xlabel("$ef$ (search)")
plt.ylabel("Recall")
save_plot("hnsw_search_ef_recall")

plt.grid(axis="y")
plt.plot(hnsw_ef_search.df["ef"], hnsw_ef_search.df["time_ms"])
plt.scatter(hnsw_ef_search.df["ef"], hnsw_ef_search.df["time_ms"])
plt.xlabel("$ef$ (search)")
plt.ylabel("Search time (ms)")
save_plot("hnsw_search_ef_search_time")

rnn_start_candidates = thesis_exp("exp_rnn_effect_of_start_candidates")
rnn_start_candidates.print()

# plt.grid(axis="y")
# plt.plot([1,2,3,4,5,6,7], rnn_start_candidates.df["recall"])
# plt.scatter([1,2,3,4,5,6,7], rnn_start_candidates.df["recall"])
# plt.xticks(ticks=[1,2,3,4,5,6,7], labels =[1,2,4,8,16,32,64])
# plt.yticks([i/100 for i in range(95,99,1)])
# plt.xlabel("Start candidates")
# plt.ylabel("Recall")
# save_plot("rnn_start_candidates_recall")

# plt.grid(axis="y")
# plt.plot([1,2,3,4,5,6,7], rnn_start_candidates.df["time_ms"])
# plt.scatter([1,2,3,4,5,6,7], rnn_start_candidates.df["time_ms"])
# plt.xticks( ticks=[1,2,3,4,5,6,7], labels =[1,2,4,8,16,32,64])
# plt.xlabel("Start candidates")
# plt.ylabel("Search time (ms)")
# save_plot("rnn_start_candidates_search_time")

set_page_fract(3)


hnsw_n = thesis_exp("exp_hnsw_effect_of_n").sort_by("n").with_build_ms_per_n()
hnsw_n.df["n_rank"] = hnsw_n.df["n"].rank(ascending=True).astype(int)
hnsw_n.print()
triptychon_plot_n_10k_to_10m(hnsw_n.df, "hnsw_n")

ensemble_n = thesis_exp("exp_ensemple_effect_of_n_").sort_by("n").with_build_ms_per_n()
ensemble_n.df["n_rank"] = ensemble_n.df["n"].rank(ascending=True).astype(int)
ensemble_n.print()
triptychon_plot_n_10k_to_10m(ensemble_n.df, "ensemble_n")

rnn_n = thesis_exp("exp_rnn_effect_of_n_").sort_by("n").with_build_ms_per_n()
rnn_n.df["n_rank"] = rnn_n.df["n"].rank(ascending=True).astype(int)
rnn_n.print()
triptychon_plot_n_10k_to_10m(rnn_n.df, "rnn_n")

def extract_10k_vs_10m_info(df: pd.DataFrame, name: str) -> dict:
    res = {"name": name}
    
    for field in ["recall", "build_ms_per_n", "time_ms"]:
        val_10k = df[field].iloc[0]
        val_10m = df[field].iloc[-1]
        factor = val_10m / val_10k
        field_fmt = "{:.3f}" # "{:.3f}" if field == "recall" else "{:.2f}ms"
        res[f"{field}_10k"] = field_fmt.format(val_10k)
        res[f"{field}_10m"] = field_fmt.format(val_10m)
        res[f"{field}_factor"] = "x{:.2f}".format(factor)
    return res
compare_10k_vs_10m = pd.DataFrame([extract_10k_vs_10m_info(hnsw_n.df, "HNSW"), extract_10k_vs_10m_info(rnn_n.df, "RNN-Descent"), extract_10k_vs_10m_info(ensemble_n.df, "VP-Tree Ensemble")])
print("compare_10k_vs_10m :")
df_print(compare_10k_vs_10m)
compare_10k_vs_10m.to_latex("tables/compare_10k_vs_10m.tex", index=False)

hnsw_m_max = thesis_exp("exp_hnsw_effect_of_m_max")
hnsw_m_max.print()
triptychon_plot(hnsw_m_max.df, "m_max", "\emph{m\_max}", "hnsw_m_max", recall_min = 75, recall_step = 5)

hnsw_ef_construction = thesis_exp("exp_hnsw_effect_of_ef_construction")
hnsw_ef_construction.print()
triptychon_plot(hnsw_ef_construction.df, "ef_construction", "\emph{ef\_construction}", "hnsw_ef_construction", recall_min = 80, recall_step = 5)

hnsw_level_norm = thesis_exp("exp_hnsw_effect_of_level_norm")
hnsw_level_norm.print()
triptychon_plot(hnsw_level_norm.df, "level_norm_param", "\emph{level\_norm}", "hnsw_level_norm", recall_min = 93, recall_max=96, recall_step = 1)

hnsw_k = thesis_exp("exp_hnsw_effect_of_ef_search", "hnsw_k").filter_col_eq("ef", "k")
hnsw_k.print()

rnn_ef_search = thesis_exp("exp_rnn_effect_of_ef_search", "rnn_ef_search").filter("k", 30)
rnn_ef_search.print()

rnn_k = thesis_exp("exp_rnn_effect_of_ef_search", "rnn_k").filter_col_eq("ef", "k")
rnn_k.print()

rnn_inner_loops = thesis_exp("exp_rnn_effect_of_inner", "rnn_inner_loops").take(20)
rnn_inner_loops.print()
triptychon_plot(rnn_inner_loops.df, "inner_loops", "\emph{t\_inner}", "rnn_inner_loops", recall_min = 20, recall_max=100, recall_step=20, build_time_min=0, search_time_min=0)

rnn_outer_loops = thesis_exp("exp_rnn_effect_of_outer", "rnn_outer_loops")
rnn_outer_loops.print()
# just used as table for now, no graph needed




stitching_fraction = thesis_exp("exp_stitching_effect_of_fraction")
stitching_fraction.print()
# remap the strings in the stich_mode column to shorter strings:
stitch_method_remap = {
    "RandomNegToPosCenterAndBack": "method 1",
    "RandomNegToRandomPosAndBack": "method 2",
    "DontStarveXXSearch": "method 3",
    "MultiEf": "method 4"
}
method_colors = {
    "method 1": "#d62728",
    "method 2": "#2ca02c",
    "method 3": "#ff7f0e",
    "method 4": "#1f77b4"
}
stitching_fraction.df["stitch_mode"] = stitching_fraction.df["stitch_mode"].map(stitch_method_remap)

set_page_fract(1)
# plot fraction against recall with group by stitch_mode:
plt.grid(axis="y")
for stitch_mode in stitching_fraction.df["stitch_mode"].unique():
    subdf = stitching_fraction.df[stitching_fraction.df["stitch_mode"] == stitch_mode]
    plt.plot(subdf["neg_fraction"], subdf["recall"], color = method_colors[stitch_mode])
    plt.scatter(subdf["neg_fraction"], subdf["recall"], label=stitch_mode, color = method_colors[stitch_mode])
plt.xlabel("Fraction of negative half")
plt.ylabel("Recall")
plt.legend()
save_plot( "stitching_fraction_recall")

# plt.grid(axis="y")
# for stitch_mode in stitching_fraction.df["stitch_mode"].unique():
#     subdf = stitching_fraction.df[stitching_fraction.df["stitch_mode"] == stitch_mode]
#     plt.plot(subdf["neg_fraction"], subdf["build_ms"]/1000, color = method_colors[stitch_mode])
#     plt.scatter(subdf["neg_fraction"], subdf["build_ms"]/1000, label=stitch_mode, color = method_colors[stitch_mode])
# plt.xlabel("Fraction of negative half")
# plt.ylabel("Build time (s)")
# plt.legend()
# save_plot("stitching_fraction_build_time")

# plt.grid(axis="y")
# for stitch_mode in stitching_fraction.df["stitch_mode"].unique():
#     subdf = stitching_fraction.df[stitching_fraction.df["stitch_mode"] == stitch_mode]
#     plt.plot(subdf["neg_fraction"], subdf["time_ms"], color = method_colors[stitch_mode])
#     plt.scatter(subdf["neg_fraction"], subdf["time_ms"], label=stitch_mode, color = method_colors[stitch_mode])
# plt.xlabel("Fraction of negative half")
# plt.ylabel("Search time (ms)")
# plt.legend()
# save_plot("stitching_fraction_search_time")

stitching_chunk_size = thesis_exp("exp_stitching_effect_of_max_chunk_size")
stitching_chunk_size.df["stitch_mode"] = stitching_chunk_size.df["stitch_mode"].map(stitch_method_remap)
stitching_chunk_size.df = stitching_chunk_size.df[stitching_chunk_size.df["stitch_mode"] != "method 1"]
stitching_chunk_size.print()
stitching_chunk_size.df["max_chunk_size"] = stitching_chunk_size.df["max_chunk_size"].map({64: 1, 128: 2, 256: 3, 512: 4, 1024: 5})
ticks = [1,2,3,4,5]
tick_labels = [64,128,256,512,1024]

# plot 3 graphs next to each other, one for recall, one for build time, one for search time

fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.5))
fig.tight_layout()
fig.subplots_adjust(wspace=0.4)
# plot fraction against recall with group by stitch_mode:
axs[0].grid(axis="y")
for stitch_mode in stitching_chunk_size.df["stitch_mode"].unique():
    subdf = stitching_chunk_size.df[stitching_chunk_size.df["stitch_mode"] == stitch_mode]
    axs[0].plot(subdf["max_chunk_size"], subdf["recall"], color = method_colors[stitch_mode])
    axs[0].scatter(subdf["max_chunk_size"], subdf["recall"], label=stitch_mode, color = method_colors[stitch_mode])
    axs[0].set_xlabel("Maximum chunk size")
    axs[0].set_ylabel("Recall")
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(tick_labels)
# plot fraction against build time with group by stitch_mode:
axs[1].grid(axis="y")
for stitch_mode in stitching_chunk_size.df["stitch_mode"].unique():
    subdf = stitching_chunk_size.df[stitching_chunk_size.df["stitch_mode"] == stitch_mode]
    axs[1].plot(subdf["max_chunk_size"], subdf["build_ms"]/1000, color = method_colors[stitch_mode])
    axs[1].scatter(subdf["max_chunk_size"], subdf["build_ms"]/1000, label=stitch_mode, color = method_colors[stitch_mode])
    axs[1].set_xlabel("Maximum chunk size")
    axs[1].set_ylabel("Build time (s)")
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(tick_labels)
# plot fraction against search time with group by stitch_mode:
axs[2].grid(axis="y")
for stitch_mode in stitching_chunk_size.df["stitch_mode"].unique():
    subdf = stitching_chunk_size.df[stitching_chunk_size.df["stitch_mode"] == stitch_mode]
    axs[2].plot(subdf["max_chunk_size"], subdf["time_ms"], color = method_colors[stitch_mode])
    axs[2].scatter(subdf["max_chunk_size"], subdf["time_ms"], label=stitch_mode, color = method_colors[stitch_mode])
    axs[2].set_xlabel("Maximum chunk size")
    axs[2].set_ylabel("Search time (ms)")
    axs[2].set_xticks(ticks)
    axs[2].set_xticklabels(tick_labels)
# add a legend on top of the 3 plots with the stitch_mode colors (remove the duplicates)
handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
# add more padding to top of figure such that legend does not overlap plots:
plt.subplots_adjust(top=0.8)
save_plot("stitching_chunk_size", tight = False)

stitching_m_max = thesis_exp("exp_stitching_effect_of_m_max")
stitching_m_max.df["stitch_mode"] = stitching_m_max.df["stitch_mode"].map(stitch_method_remap)
stitching_m_max.df = stitching_m_max.df[stitching_m_max.df["stitch_mode"] != "method 1"]
stitching_m_max.print()

fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.5))
fig.tight_layout()
fig.subplots_adjust(wspace=0.4)
# plot fraction against recall with group by stitch_mode:
axs[0].grid(axis="y")
for stitch_mode in stitching_m_max.df["stitch_mode"].unique():
    subdf = stitching_m_max.df[stitching_m_max.df["stitch_mode"] == stitch_mode]
    axs[0].plot(subdf["m_max"], subdf["recall"], color = method_colors[stitch_mode])
    axs[0].scatter(subdf["m_max"], subdf["recall"], label=stitch_mode, color = method_colors[stitch_mode])
    axs[0].set_xlabel("\emph{m\_max}")
    axs[0].set_ylabel("Recall")
    axs[0].set_xticks(range(0,41, 8))
# plot fraction against build time with group by stitch_mode:
axs[1].grid(axis="y")
for stitch_mode in stitching_m_max.df["stitch_mode"].unique():
    subdf = stitching_m_max.df[stitching_m_max.df["stitch_mode"] == stitch_mode]
    axs[1].plot(subdf["m_max"], subdf["build_ms"]/1000, color = method_colors[stitch_mode])
    axs[1].scatter(subdf["m_max"], subdf["build_ms"]/1000, label=stitch_mode, color = method_colors[stitch_mode])
    axs[1].set_xlabel("\emph{m\_max}")
    axs[1].set_ylabel("Build time (s)")
    axs[1].set_xticks(range(0,41, 8))
# plot fraction against search time with group by stitch_mode:
axs[2].grid(axis="y")
for stitch_mode in stitching_m_max.df["stitch_mode"].unique():
    subdf = stitching_m_max.df[stitching_m_max.df["stitch_mode"] == stitch_mode]
    axs[2].plot(subdf["m_max"], subdf["time_ms"], color = method_colors[stitch_mode])
    axs[2].scatter(subdf["m_max"], subdf["time_ms"], label=stitch_mode, color = method_colors[stitch_mode])
    axs[2].set_xlabel("\emph{m\_max}")
    axs[2].set_ylabel("Search time (ms)")
    axs[2].set_xticks(range(0,41, 8))
# add a legend on top of the 3 plots with the stitch_mode colors (remove the duplicates)
handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
# add more padding to top of figure such that legend does not overlap plots:
plt.subplots_adjust(top=0.8)
save_plot("stitching_m_max", tight = False)

stitching_same_chunk_m_max = thesis_exp("exp_stitching_effect_of_same_chunk_m_max")
stitching_same_chunk_m_max.print()
triptychon_plot(stitching_same_chunk_m_max.df, "same_chunk_m_max", "\emph{same\_chunk\_m\_max}", "stitching_same_chunk_m_max", 
                recall_max=100, recall_min=50, recall_step=10, x_min=0)

ensemble_n_vp_trees = thesis_exp("exp_ensemble_effect_of_n_vp_trees")
ensemble_n_vp_trees.print()
# triptychon_plot(ensemble_n_vp_trees.df, "n_vp_trees", "\\texttt{n\_vp\_trees}", "ensemble_n_vp_trees")

ensemble_m_max = thesis_exp("exp_ensemble_effect_of_m_max")
ensemble_m_max.print()
# triptychon_plot(ensemble_m_max.df, "m_max", "\\texttt{m\_max}", "ensemble_m_max", recall_min= 40, recall_step=10)

ensemble_chunk_size = thesis_exp("exp_ensemble_effect_of_chunk_size")
ensemble_chunk_size.print()
# triptychon_plot(ensemble_chunk_size.df, "max_chunk_size", "maximum chunk size", "ensemble_chunk_size", recall_min= 80, recall_step=10)

ensemble_same_chunk_m_max = thesis_exp("exp_ensemble_effect_of_same_chunk_m_max")
ensemble_same_chunk_m_max.print()
triptychon_plot(ensemble_same_chunk_m_max.df, "same_chunk_m_max", "\emph{same\_chunk\_m\_max}", "ensemble_same_chunk_m_max", 
                recall_min  = 85, recall_step = 5, recall_max=95, x_min=0, build_time_min=0.65, build_time_max=0.8)

ensemble_level_norm = thesis_exp("exp_ensemble_effect_of_level_norm")
ensemble_level_norm.print()
triptychon_plot(ensemble_level_norm.df, "level_norm", "\\texttt{level\_norm}", "ensemble_level_norm")

ensemble_rnn = thesis_exp("exp_ensemble_effect_of_brute_force_vs_rnn_smmax20").sort_by(["strategy", "o_loops", "max_chunk_size"])
ensemble_rnn.df = ensemble_rnn.df.sort_values(by=[])
def row_to_strat_str(row: pd.Series) -> str:
    if row["strategy"] == "RNNDescent":
        return f"RNN-Descent {int(row['o_loops'])}x{int(row['i_loops'])}"
    else:
        return f"Brute Force"
strat_colors = {
    "Brute Force": "#2ca02c",
    "RNN-Descent 1x3":  "#ff7f0e" ,
    "RNN-Descent 2x3": "#d62728",
    # "RNN-Descent 3x3": "#1f77b4"
}
strats = [ "RNN-Descent 1x3", "RNN-Descent 2x3", "Brute Force",]
ensemble_rnn.df["strat"] = ensemble_rnn.df.apply(row_to_strat_str, axis=1)
ensemble_rnn.df = ensemble_rnn.df[ensemble_rnn.df["strat"] != "RNN-Descent 3x3"] # adds no values, same curve as 2x3
ensemble_rnn.df["max_chunk_size"] = ensemble_rnn.df["max_chunk_size"].map({64: 1,128:2,256: 3,512: 4,1024: 5,2048: 6})
max_chunk_size_labels = [64,128,256,512,1024,2048]
ensemble_rnn.print()
fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.5))
fig.tight_layout()
fig.subplots_adjust(wspace=0.3)
axs[0].grid(axis="y")
for strat in strats:
    subdf = ensemble_rnn.df[ensemble_rnn.df["strat"] == strat]
    axs[0].plot(subdf["max_chunk_size"], subdf["recall"], color = strat_colors[strat])
    axs[0].scatter(subdf["max_chunk_size"], subdf["recall"], label=strat, color = strat_colors[strat])
    axs[0].set_xlabel("Maximum chunk size")
    axs[0].set_ylabel("Recall")
    axs[0].set_xticks(range(1,7))
    axs[0].set_xticklabels(max_chunk_size_labels)
    axs[0].set_yticks([0.80,0.85,0.90,0.95,1.0])
axs[1].grid(axis="y")
for strat in strats:
    subdf = ensemble_rnn.df[ensemble_rnn.df["strat"] == strat]
    axs[1].plot(subdf["max_chunk_size"], subdf["build_ms"]/1000, color = strat_colors[strat])
    axs[1].scatter(subdf["max_chunk_size"], subdf["build_ms"]/1000, label=strat, color = strat_colors[strat])
    axs[1].set_xlabel("Maximum chunk size")
    axs[1].set_ylabel("Build time (s)")
    axs[1].set_xticks(range(1,7))
    axs[1].set_xticklabels(max_chunk_size_labels)
    axs[1].set_yticks([0,20,40,60,80])
handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
plt.subplots_adjust(top=0.8) # add more padding to top of figure such that legend does not overlap plots
save_plot("ensemble_brute_force", tight = False)


a_10m = thesis_exp("a_10m_multi_threaded")
# bestpicked.sort_by(["model", "max_chunk_size", "inner_loops", "outer_loops","ef"])
a_10m.df.rename(columns={"search_recall": "recall"}, inplace=True)
a_10m.print()

fig, axs = plt.subplots(1, 3, figsize=(16.8, 8.5))

# # plot the ef on the x axis and the recall on the y axis, with different lines for the different models (model by key)
# axs[0].grid(axis="y")
# for par in a_10m.df["params"].unique():
#     subdf = a_10m.df[a_10m.df["params"] == par]
#     axs[0].plot(subdf["ef"], subdf["recall"], label=par)
#     axs[0].scatter(subdf["ef"], subdf["recall"])
# axs[0].set_xlabel("ef")
# axs[0].set_ylabel("Recall")

# # plot the ef on the x axis and the build time on the y axis, with different lines for the different models (model by key)
# axs[1].grid(axis="y")
# for par in a_10m.df["params"].unique():
#     subdf = a_10m.df[a_10m.df["params"] == par]
#     axs[1].plot(subdf["ef"], subdf["search_ms"], label=par)
#     axs[1].scatter(subdf["ef"], subdf["search_ms"])
# axs[1].set_xlabel("ef")
# axs[1].set_ylabel("Search time (ms)")

# axs[2].grid(axis="y")
# for par in a_10m.df["params"].unique():
#     subdf = a_10m.df[a_10m.df["params"] == par]
#     axs[2].plot(subdf["ef"], subdf["build_secs"], label=par)
#     axs[2].scatter(subdf["ef"], subdf["build_secs"])
# axs[2].set_xlabel("ef")
# axs[2].set_ylabel("Build time (s)")

# # add a legend on top of the 2 plots with the key colors (remove the duplicates)
# handles, labels = axs[0].get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
# # add more padding to top of figure such that legend does not overlap plots:
# plt.subplots_adjust(top=0.8)
# save_plot("a_10m_multi", tight = False)




# ensemble_multiple_vantage_points = thesis_exp("exp_ensemble_effect_of_multiple_vantage_points")
# ensemble_multiple_vantage_points.print()

# compare_25 = thesis_exp("experiment2024-09-25")
# compare_25.print()

# compare_29= thesis_exp("experiment2024-09-29")
# compare_29.print()

# compare_search_ef = thesis_exp("experiment_compare_search_ef")
# compare_search_ef.df = compare_search_ef.df.sort_values(by=["model", "max_chunk_size", "ef"])
# compare_search_ef.print()

# compare_search_ef_single = thesis_exp("experiment_compare_search_ef_single_threaded")
# compare_search_ef_single.df = compare_search_ef_single.df.sort_values(by=["model", "max_chunk_size", "ef"])
# compare_search_ef_single.print()

# save_all_experiments_as_latex_tables()

