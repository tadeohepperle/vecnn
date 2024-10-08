"""
This module provides a class Experiment that can be constructed from
one or more csv files. It should do a few things:
- first it creates more columns, including the metadata kv pairs in the csv files filename. Also splits the params into individual columns.
- then it removes columns where all values are the same and put them into a metadata dict.

This helps us to extract the actual useful information from all the info spit out by the compare.rs Rust script.
"""

import matplotlib.pyplot as plt

from typing import List, Tuple
import pandas as pd
import os
import re


class Experiment:
    name: str
    csv_paths: List[str]
    df: pd.DataFrame
    common: dict

    def __init__(self, csv_path_prefix: str, name: str | None = None):
        self.csv_paths = find_paths_with_prefix(csv_path_prefix)
        if name == None:
            self.name = self.csv_paths[0].split("/")[-1].split(".csv")[0]
        else:
            self.name = name
        all_rows = [] # list of dicts

        # flatten out structure into individual rows:
        for path in self.csv_paths:
            df = pd.read_csv(path, skip_blank_lines=True, comment="#")
            meta_params = extract_params_from_path(path)
            for index, row in df.iterrows():
                row_dict = row.to_dict()
                split_params_column(row_dict)
                split_search_params_column(row_dict)
                for k, v in meta_params.items():
                    row_dict[k] = v
                all_rows.append(row_dict)

        # convert all of the rows to a pandas dataframe
        df = pd.DataFrame(all_rows)
        common = {} 
        # extract all rows where each value is the same into a common dict
        for col in df.columns:
            if df[col].nunique(dropna=False) == 1:
                common[col] = df[col].iloc[0]
                del df[col]
        df.rename(lambda x: x if not x.endswith("_mean") else x[:-5], axis=1, inplace=True)

        self.common = common
        self.df = df
        global ALL_EXPERIMENTS
        ALL_EXPERIMENTS.append(self)
        pass

    def print(self, with_params_col: bool = False):
        print("#" * 80)
        print("Name: ", self.name)
        print("Common: ", self.common)
        print("Data: ")
        with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000, "display.width",1000):
            if with_params_col or "params" not in self.df.columns:
                print(self.df.to_string(index=False))
            else:   
                print(self.df.drop(columns= ["params"]).to_string(index=False))
        print("#" * 80)


    def sort_by(self, col: str):
        self.df = self.df.sort_values(by=col)
        return self

    def filter(self, col, value):
        self.df = self.df[self.df[col] == value]
        return self
    
    def filter_col_eq(self, col1, col2):
        self.df = self.df[self.df[col1] == self.df[col2]]
        return self
    
    def discard_cols(self, cols: List[str]):
        for col in cols:
            del self.df[col]
        return self
    
    def take(self, n: int):
        self.df = self.df.head(n)
        return self
    
    def print_latex(self, columns = None, decimal_digits=3):
        print(self.latex_str(columns, decimal_digits))


    def with_build_ms_per_n(self):
        assert "n" not in self.common
        assert "n" in self.df.columns
        self.df["build_ms_per_n"] = self.df["build_ms"] / self.df["n"]
        return self

    def latex_str(self, columns = None, decimal_digits=3):
        if columns != None:
            df = self.df[columns]
        else:
            df = self.df
        caption = str(self.common)
        caption = "experiment: " + caption.replace("{", "").replace("}", "").replace("'", "").replace("_", "\_").replace(": ", "=")
        latex_str = df.to_latex(index=False, caption=caption, label=self.name, float_format=f"%.{decimal_digits}f")
        return latex_str

    def plot(self, x_col: str, y_col: str, x_label = None, y_label = None, save_path: str = None, show: bool = True, log_x: bool = False):
        if x_label == None:
            x_label = x_col
        if y_label == None:
            y_label = y_col
        plt.plot(self.df[x_col], self.df[y_col])
        # make x axis logarithmic
        # draw points on line:
        plt.scatter(self.df[x_col], self.df[y_col])
        if log_x:
            plt.xscale("log")
        # show the ticks on y axis up to including 1.0
        plt.yticks([i/10 for i in range(11)])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        if save_path != None:
            plt.savefig(save_path)
        if show:
            plt.show()
        return self

ALL_EXPERIMENTS: List[Experiment] =[]

# returns all paths to files with the given prefix
def find_paths_with_prefix(prefix: str) -> List[str]:
    directory = os.path.dirname(prefix)
    prefix_name = os.path.basename(prefix)
    
    return [directory + "/" + f for f in os.listdir(directory) if f.startswith(prefix_name) and f.endswith(".csv")]

def split_params_column(row_dict):
    if "params" not in row_dict:
        return
    params_str = row_dict["params"]
    params_str = re.sub(
        r'strategy: RNNDescent\s*{\s*o_loops:\s*(\d+),\s*i_loops:\s*(\d+)\s*}', 
        r'strategy: RNNDescent, o_loops: \1, i_loops: \2', 
        params_str
    )
    # del row_dict["params"]
    s =  params_str.split("{")
    
    row_dict["model"] = s[0].strip()
    
    rest = s[1].split("}")[0].strip()
    for kv in rest.split(", "):
        k, v = kv.split(": ")
        v = str_to_float_or_int(v)
        row_dict[k] = v
    pass
def split_search_params_column(row_dict):
    if "search_params" not in row_dict:
        return
    parts = row_dict["search_params"].split(" ")
    del row_dict["search_params"]
    for part in parts:
        k, v = part.split("=")
        v = str_to_float_or_int(v)
        row_dict[k] = v
    pass

def str_to_float_or_int(s: str):
    return int(s) if s.isdigit() else float(s) if s.replace('.', '', 1).isdigit() else s

def extract_params_from_path(file_name: str) -> dict:
    try:
        split = file_name.split("n=")
        n_str = split[1].split("_queries_")[0]
        n_queries_str = split[2].split(".csv")[0]
        return {"n": int(n_str), "n_queries": int(n_queries_str)}
    except Exception as e:
        # print(file_name, e)
        return {"n": 10120191, "n_queries": 10000}


def save_all_experiments_as_latex_tables(path: str = "latex_tables.txt"):
    s = ""
    for exp in ALL_EXPERIMENTS:
        s+="\n\n\n"
        name = exp.name.replace("_", "-")
        s+="\n"
        s+=exp.latex_str()
    with open(path, "w") as f:
        f.write(s)


# PATH1 = "../vecnn/experiments/hnsw_effect_of_ef_construction_n=1000_queries_n=100.csv"
# PATH2 = "../vecnn/experiments/stitching_n_candidates_n=200000_queries_n=100.csv"

# Experiment([PATH1]).print()
