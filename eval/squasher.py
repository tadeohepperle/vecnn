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

class Experiment:
    name: str
    csv_paths: List[str]
    df: pd.DataFrame
    common: dict

    def __init__(self, csv_paths: List[str], name: str | None = None):
        if name == None:
            self.name = csv_paths[0].split("/")[-1].split(".csv")[0]
        else:
            self.name = name
        all_rows = [] # list of dicts

        # flatten out structure into individual rows:
        for path in csv_paths:
            df = pd.read_csv(path)
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

        self.csv_paths = csv_paths
        self.common = common
        self.df = df
        pass

    def print(self):
        print("Common: ", self.common)
        with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000, "display.width",1000):
            print(self.df)

    def filter(self, col, value):
        self.df = self.df[self.df[col] == value]
        return self
    
    def print_latex(self, columns = None):
        if columns != None:
            df = self.df[columns]
        else:
            df = self.df
        caption = str(self.common)
        caption = "experiment: " + caption.replace("{", "").replace("}", "").replace("'", "").replace("_", "\_").replace(": ", "=")
        latex_str = df.to_latex(index=False, caption=caption, label=self.name)
        print(latex_str)

    def plot(self, x_col: str, y_col: str, x_label = None, y_label = None, save_path: str = None, show: bool = True, log_x: bool = True, log_y: bool = False):
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
        if log_y:
            plt.yscale("log")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        
        if save_path:
            plt.savefig(save_path)
        return self

def split_params_column(row_dict):
    if "params" not in row_dict:
        return
    params_str = row_dict["params"]
    del row_dict["params"]
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
        print(e)
        return {}

# PATH1 = "../vecnn/experiments/hnsw_effect_of_ef_construction_n=1000_queries_n=100.csv"
# PATH2 = "../vecnn/experiments/stitching_n_candidates_n=200000_queries_n=100.csv"

# Experiment([PATH1]).print()