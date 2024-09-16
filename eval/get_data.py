import pandas as pd
import numpy as np
from typing import Any, Tuple
import sys
import urllib.request
import os
import laion_util

LAION_100M_URL = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=100M.h5"
LAION_10M_URL = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=10M.h5"
LAION_300K_URL = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=300K.h5"
LAION_10K_QUERIES_URL = "http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5"

LAION_100M_FILE_NAME = "laion2B-en-clip768v2-n=100M.h5"
LAION_10M_FILE_NAME = "laion2B-en-clip768v2-n=10M.h5"
LAION_300K_FILE_NAME = "laion2B-en-clip768v2-n=300K.h5"
LAION_10K_QUERIES_FILE_NAME = "public-queries-2024-laion2B-en-clip768v2-n=10k.h5"

def download_file(url: str, file_name: str):
    print(f"Downloading {url} to {file_name}")
    with urllib.request.urlopen(url) as response:
        total_size = int(response.getheader('Content-Length').strip())
        bytes_downloaded = 0
        chunk_size = 2**18 # 256 KB

        last_percent_complete = 0
    
        with open(file_name, 'wb') as out_file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                bytes_downloaded += len(chunk)
                
                # Display progress
                percent_complete = (bytes_downloaded / total_size) * 100
                if last_percent_complete + 1 < percent_complete:
                    last_percent_complete = percent_complete
                    print(f"Downloaded {bytes_downloaded} of {total_size} bytes ({percent_complete:.0f}%)")

def clear_dir(data_dir: str):
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            os.remove(os.path.join(data_dir, f))
    else:
        os.makedirs(data_dir)


if len(sys.argv) <2:

    print("Usage: get_data.py <data_dir> [clear] [convert] [300k] [10m] [100m]")
    print("    data_dir: directory to download the data to")
    print("    clear: clear the data directory before downloading")
    print("    convert: convert the h5 files to binary for easier reading from rust")
    print("    300k: include the 300k dataset")
    print("    10m: include the 10m dataset")
    print("    100m: include the 100m dataset")
    print("Example: get_data.py ./my_data_dir clear convert 300k 10m 100m")
    print("Note: the 10k queries are always downloaded")
    sys.exit(1)

data_dir = sys.argv[1]

options = sys.argv[2:]

if "clear" in options:
    clear_dir(data_dir)
    print(f"Cleared directory {data_dir}")

laion_10k_queries_path = data_dir + "/" + LAION_10K_QUERIES_FILE_NAME
download_file(LAION_10K_QUERIES_URL, laion_10k_queries_path)
print("Downloaded 10K queries to "+ laion_10k_queries_path)
if "convert" in options:
    laion_util.convert_h5_emb_to_binary(laion_10k_queries_path, data_dir + "/laion_10k_queries")
    print("Converted 10K queries")

if "300k" in options:
    laion_300k_path = data_dir + "/" + LAION_300K_FILE_NAME
    download_file(LAION_300K_URL, laion_300k_path)
    print("Downloaded 300K dataset to "+ laion_300k_path)
    if "convert" in options:
        laion_util.convert_h5_emb_to_binary(laion_300k_path, data_dir + "/laion_300k")
        print("Downloaded 300K dataset")



if "10m" in options or "10M" in options:
    laion_10m_path = data_dir + "/" + LAION_10M_FILE_NAME
    download_file(LAION_10M_URL, laion_10m_path)
    print("Downloaded 10M dataset to " + laion_10m_path)
    if "convert" in options:
        laion_util.convert_h5_emb_to_binary(laion_10m_path, data_dir + "/laion_10m")
        print("Converted 10M dataset")

if "100m" in options or "100M" in options:
    laion_100m_path = data_dir + "/" + LAION_100M_FILE_NAME
    download_file(LAION_100M_URL, laion_100m_path)
    print("Downloaded 100M dataset to " + laion_100m_path)
    if "convert" in options:
        laion_util.convert_h5_emb_to_binary(laion_100m_path, data_dir + "/laion_100m")
        print("Converted 100M dataset")


# print all files that the data dir contains
print(f"Contents of directory {data_dir}:")
for f in os.listdir(data_dir):
    f_path = os.path.join(data_dir, f)
    size = os.path.getsize(f_path)
    if size > 1024**3:
        size = f"{size/1024**3:.2f}GB"
    elif size > 1024**2:
        size = f"{size/1024**2:.2f}MB"
    elif size > 1024:
        size = f"{size/1024:.2f}KB"
    else:
        size = f"{size}B"
        
    print(f"    {size.ljust(10)} {f_path}")