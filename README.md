ToMATo for protein conformation
==============================

Analyze protein conformations using the topology-based method ToMATo to cluster the conformations

<b>Important</b>: Please find the report in `./reports/tomato.pdf`

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebook for exploring the data using the code in ./src
    │
    ├── reports            <- LaTeX file with the final report.
    │   └── figures        <- Generated graphics and figures to be used in the report.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │
    └── src                <- Source code made for in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download and pre-process data. Output files should be in ./data/raw
        │   └── get_raw_data.sh
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        └── models         <- Scripts to train models
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
     

How to execute the code
--------
1. You should be using an environment with python3.9

2. Inside your environment, install the requirements by using:

```shell
pip install -r requirements.txt
```

3. To get the data go to ./src/data and execute:
```shell
./get_raw_data.sh
```

4. To get the distance matrix go to ./src/data , decide a number of proteins to work with (let's say 2000), and execute:
```shell
python3 ./build_features.py -n2000
```
For further information about how to use `build_features.py` do:
```shell
python3 ./build_features.py --help
```
The output will be saved in ./data/processed as `distances_2000prots.npy`
5. To get the mds embedding from the distance matrix, go to ./src/models and execute:
```shell
python3 ./get_embeddings.py -i '../../data/distances_2000prots.npy'
```
The output will be saved in ./data/processed as `embeddings_2000prots.npy`
6. Testing the different parameters in ToMATo was done in the Jupyter notebook `./notebooks/tomato_parameters.ipynb` 
7. Additionally, the files used to send jobs to INRIA's NEF cluster can be found in each respective folder.

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
