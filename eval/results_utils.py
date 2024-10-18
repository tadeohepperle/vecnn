from typing import Literal
import matplotlib.pyplot as plt
import pandas as pd
from squasher import Experiment, save_all_experiments_as_latex_tables

# size if 3 figures next to each other
ONE_THIRD_FONT_SIZE = 12
ONE_THIRD_FIG_SIZE = (2.8,2.5)

# size if 2 figures next to each other
ONE_HALF_FONT_SIZE = 12 # actually should be the same as ONE_THIRD_FONT_SIZE, just scale ONE_HALF_FIG_SIZE maybe???
ONE_HALF_FIG_SIZE = (4.2,3.2)

ONE_FONT_SIZE = 12 # for plots covering the entire page with
ONE_FIG_SIZE = (8.4,5.2)

plt.rcParams["font.family"] = ["Linux Libertine O", "sans-serif"]
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('axes', axisbelow=True)

def set_page_fract(one_over: Literal[1,2,3]):
    assert one_over in [1, 2, 3]
    if one_over == 1:
        plt.rcParams["font.size"] = ONE_FONT_SIZE
        plt.figure(figsize=ONE_FIG_SIZE) 
    elif one_over == 2:
        plt.rcParams["font.size"] = ONE_HALF_FONT_SIZE
        plt.figure(figsize=ONE_HALF_FIG_SIZE) 
    elif one_over == 3:
        plt.rcParams["font.size"] = ONE_THIRD_FONT_SIZE
        plt.figure(figsize=ONE_THIRD_FIG_SIZE)


def thesis_exp(prefix: str, name: str |None = None) -> Experiment:
    if name is None:
        name = prefix
    return Experiment("./thesis_experiments/" + prefix, name = name)
def save_plot(name: str, tight : bool = True):
    if tight:
        plt.tight_layout()
    plt.savefig("../../writing/images/" + name + ".pdf", bbox_inches='tight',pad_inches = 0.1, dpi = 200)
    plt.clf()
def df_print(df: pd.DataFrame):
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000, "display.width",1000, "display.max_colwidth", 1000):
        print(df)

def triptychon_plot(df: pd.DataFrame, x_col: str, x_col_label: str, experiment_name: str, recall_min: int | None  = 70, recall_max: int | None= 100, recall_step = 5, build_time_min: int |None= None, build_time_max: int |None= None,  search_time_min: int |None= None, x_min: int | None = None,):
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.5))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    axs[0].grid(axis="y")
    axs[0].plot(df[x_col], df["recall"])
    axs[0].scatter(df[x_col], df["recall"])
    axs[0].set_xlabel(x_col_label)
    axs[0].set_ylabel("Recall")
    if recall_min is not None and recall_max is not None:
        axs[0].set_yticks([i/100 for i in range(recall_min,recall_max+1,recall_step)])

    axs[1].grid(axis="y")
    axs[1].plot(df[x_col], df["build_ms"]/1000)
    if build_time_min is not None:
        axs[1].set_ylim(bottom=build_time_min)
    if build_time_max is not None:
        axs[1].set_ylim(top=build_time_max)
    axs[1].scatter(df[x_col],df["build_ms"]/1000)
    axs[1].set_xlabel(x_col_label)
    axs[1].set_ylabel("Build time (s)")

    axs[2].grid(axis="y")
    axs[2].plot(df[x_col], df["time_ms"])
    axs[2].scatter(df[x_col], df["time_ms"])
    if search_time_min is not None:
        axs[2].set_ylim(bottom=search_time_min)
    axs[2].set_xlabel(x_col_label)
    axs[2].set_ylabel("Search time (ms)")

    if x_min is not None:
        for ax in axs:
            ax.set_xlim(left=x_min)
    plt.subplots_adjust(top=0.8)
    save_plot(experiment_name, tight = False)

def triptychon_plot_n_10k_to_10m(df: pd.DataFrame, save_name: str):
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.5))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    axs[0].grid(axis="y")
    axs[0].plot(df["n_rank"], df["recall"])
    axs[0].scatter(df["n_rank"], df["recall"])
    axs[0].set_xlabel("n")
    axs[0].set_xticks(df["n_rank"][::5])
    axs[0].set_xticklabels(["10k", "100k", "1m", "10m"]) #df["n"][::5]
    axs[0].set_ylabel("Recall")
    axs[1].grid(axis="y")
    axs[1].plot(df["n_rank"], df["build_ms_per_n"])
    axs[1].scatter(df["n_rank"],df["build_ms_per_n"])
    axs[1].set_xlabel("n")
    axs[1].set_xticks(df["n_rank"][::5])
    axs[1].set_xticklabels(["10k", "100k", "1m", "10m"]) #df["n"][::5]
    axs[1].set_ylabel("Build time (ms) / point")
    axs[2].grid(axis="y")
    axs[2].plot(df["n_rank"], df["time_ms"])
    axs[2].scatter(df["n_rank"], df["time_ms"])
    axs[2].set_xlabel("n")
    axs[2].set_xticks(df["n_rank"][::5])
    axs[2].set_xticklabels(["10k", "100k", "1m", "10m"]) #df["n"][::5]
    axs[2].set_ylabel("Search time (ms)")
    plt.subplots_adjust(top=0.8)
    save_plot(save_name, tight = False)