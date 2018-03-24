<h1 align='center'> Assignment 2: Predict missing links in citation networks </h1>
<p align='center'>
<i>CentraleSupélec <br>
2017 - 2018 <hr></i></p>

__Team Name__: Les Cloue-Porte<br>
__Authors__: Adib Baziz, Reuben Dorent, Samuel Joutard, Joël Seytre<br>

_Competition_: [Predict missing links in citation networks](https://www.kaggle.com/c/ngsa-2018)<br>
Our report can be found [here](https://www.overleaf.com/14816913nhgjghdgrtfc).
## Index
1. [Code structure](#code)
2. [Features](#features)

# <a name="code"></a>Code Structure
* **bin** <br>
Contains documents that were provided from the start.
* **data** <br>
Contains the raw datasets.
* **predictors** <br>
Contains the models (each is one class) that are imported from **main.py**.<br>
Each model inherits from **NGSApredictor.py** (which is where the features are computed, the datasets are read etc.). The predictors only define the **run()** method.<br>
To create a new model simply copy the **svm_baseline.py** and modify it accordingly.
* the **main.py** file<br>
The file that should be executed. Only choose the model here and perhaps which `.csv` file you want to use for the features.<br>
Instantiates the model of your choice and runs its methods.
* the **predictor_settings.py** file <br>
This is where you can choose the setting for the run such as whether to compute features or used stored ones, what split of the training set should be used (the features were computed on 10% of the dataset).
* the **stored_xxx** files are the features that are computed by the model
    * **stored_training** for the features of the training set
    * **stored_training_labels** for the labels of the training set
    * **stored_testing** for the features of the test set

* the **requirements.txt** can easily be installed (Python 2.7) into a virtualenv environment with<br>
    * `python2 -m virtualenv ngsak-venv`
    * activate the virtualenv (`ngsak-venv\scripts\activate.bat` for Windows)
    * `pip install -r requirements.txt`<br>
*if needed (Windows...) there is the wheel for the igraph library for Python 2.7-x64*

# <a name="features"></a>Features
The features computed are:
* The three **default features** provided with the baseline <br>
    * **Title overlap**: number of overlapping words in titles
    * **Year difference** between articles
    * **Number of common authors**
* The **TF-IDF** (Term Frequency - Inverse Document Frequency) between the source and target articles:
    * **TF-IDF distance between the articles' abstracts**
    * **TF-IDF distance between the articles' author names** (probably very useless feature)
    * **TF-IDF distance between the articles' titles**
* The **number of times the target article is cited**
* The **shortest path between the source and the target articles** (discounting an existing direct edge for the training set)
    * in the **directed** graph
    * in the **undirected** copy of the graph
* The **jaccard similarity coefficients of the source and target articles**
