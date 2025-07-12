# ABCDisCoTEC



## Getting started
Downloading the code.
```
git clone ssh://git@gitlab.cern.ch:7999/cms-analysis/mlg/mlg-23-003/abcdiscotec.git
cd abcdiscotec
```


## Installing and running the code

Using a python virtual environment.

```
python -m venv --system-site-packages discotecenv
source discotecenv/bin/activate

python -m pip install --no-cache-dir pip --upgrade
python -m pip install --no-cache-dir uproot
python -m pip install --no-cache-dir pandas
python -m pip install --no-cache-dir matplotlib==3.9.3
python -m pip install --no-cache-dir torch==2.5.1
python -m pip install --no-cache-dir comet_ml
python -m pip install --no-cache-dir git+https://github.com/fleble/mdmm
python -m pip install --no-cache-dir tqdm
python -m pip install --no-cache-dir scikit-learn
python -m pip install --no-cache-dir seaborn

```

Another option is to use the `requirements.txt` file with anaconda.

Now make the toy data this example code expects
```
python make_training_data.py
```

Running the example with DoubleDiscoTEC
```
python example.py
```

Running the example with SingleDiscoTEC
```
python example_single_disco.py
```
