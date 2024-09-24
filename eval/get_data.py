from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Any, Literal, Tuple
import sys
import urllib.request
import os
from typing import Any, Tuple
import h5py


if len(sys.argv) <2:
    print("Usage: get_data.py <data_dir> [clear] [convert] [300k] [10m] [100m]")
    print("    data_dir: directory to download the data to")
    print("    clear: clear the data directory before downloading")
    print("    download: download the datasets from sisap servers")
    print("    convert: convert the h5 files to binary (slice of f32 or usize) for easier uncompressed reading from rust")
    print("    all: include all of the datasets below:")
    print("        queries: include the 10k queries dataset")
    print("        300k: include the 300k dataset")
    print("        10m: include the 10m dataset")
    print("        100m: include the 100m dataset")
    print("Example: get_data.py ./my_data_dir clear convert 300k 10m 100m")
    print("Note: the 10k queries are always downloaded")
    sys.exit(1)

data_dir = sys.argv[1]
options = sys.argv[2:]

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

@dataclass
class Source:
    name: str
    url: str
    key: Literal["emb", "knns"]

LAION_100M_URL = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=100M.h5"
LAION_10M_URL = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=10M.h5"
LAION_300K_URL = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge/laion2B-en-clip768v2-n=300K.h5"
LAION_10K_QUERIES_URL = "http://ingeotec.mx/~sadit/sisap2024-data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
LAION_10K_QUERIES_100M_GOLD_URL = "http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=100M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
LAION_10K_QUERIES_10M_GOLD_URL = "http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
LAION_10K_QUERIES_300K_GOLD_URL = "http://ingeotec.mx/~sadit/sisap2024-data/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
SOURCES = [
    Source("queries", LAION_10K_QUERIES_URL, "emb"),
    Source("300k", LAION_300K_URL, "emb"),
    Source("10m", LAION_10M_URL, "emb"),
    Source("100m", LAION_100M_URL, "emb"),
    Source("gold_300k", LAION_10K_QUERIES_300K_GOLD_URL, "knns"),
    Source("gold_10m", LAION_10K_QUERIES_10M_GOLD_URL, "knns"),
    Source("gold_100m", LAION_10K_QUERIES_100M_GOLD_URL, "knns"),
]



OPTION_ALL = "all" in options
OPTION_CLEAR = "clear" in options
OPTION_CONVERT = "convert" in options
OPTION_DOWNLOAD = "download" in options

if OPTION_CLEAR:
            clear_dir(data_dir)

for source in SOURCES:
    source_h5_path = data_dir + "/laion_" + source.name + ".h5"
    if source.name in options or OPTION_ALL:
        if OPTION_DOWNLOAD:
            download_file(source.url, source_h5_path)
        print(f"Downloaded {source.name} to {source_h5_path}")
        if OPTION_CONVERT:
            f = h5py.File(source_h5_path, 'r')
            assert source.key == "emb" or source.key == "knns"
            dtype = "float32" if source.key == "emb" else "uint64"
            data = np.array(f[source.key]).astype(dtype)
            if source.key == "knns":
                data = np.subtract(data, 1) # for some reason the gold standard knn indices from sisap website are 1-based.
            bin_path = f"{data_dir}/laion_{source.name}_{data.shape}.bin"
            data.tofile(bin_path)
            print(f"Converted {source_h5_path} to {bin_path}")

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