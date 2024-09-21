Upon updating the vecnnpy python package wrapper around vecnn, run:

```sh
python -m pip install -e ../vecnnpy
```

Some useful commands on the server:

```sh
# if data missing:
./eval/.venv/bin/python ./eval/get_data.py /data/hepperle clear convert 300k 10m
cargo run --bin compare --release &> out.txt
.venv/bin/python -m pip install -e ../vecnnpy
.venv/bin/python eval.py
```
