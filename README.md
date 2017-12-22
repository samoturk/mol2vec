# Mol2vec
[Mol2vec](http://pubs.acs.org/doi/10.1021/acs.jcim.7b00616) - an unsupervised machine learning approach to learn vector representations of molecular substructures

[![Documentation Status](https://readthedocs.org/projects/mol2vec/badge/?version=latest)](http://mol2vec.readthedocs.org/en/latest/)

## Requirements
* **Python 3** (Python 2.x is [not supported](http://www.python3statement.org/))
* [NumPy](http://www.numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [pandas](http://pandas.pydata.org/)
* [IPython](https://ipython.org/)
* [RDKit](http://www.rdkit.org/docs/Install.html)
* [scikit-learn](http://scikit-learn.org/stable/)
* [gensim](https://radimrehurek.com/gensim/)
* [tqdm](https://pypi.python.org/pypi/tqdm)
* [joblib](https://pythonhosted.org/joblib/)

## Installation
`pip install git+https://github.com/samoturk/mol2vec`

#### Documentation
Read the documentation on [Read the Docs](http://mol2vec.readthedocs.io/en/latest/).

To build the documentation install `sphinx`, `numpydoc` and `sphinx_rtd_theme` and then run `make html` in `docs` directory.

## Usage
### As python module
```python
from mol2vec import features
from mol2vec import helpers
```
First line imports functions to generate "sentences" from molecules and train the model, and second line imports functions useful for depictions. 
Check [examples](https://github.com/samoturk/mol2vec/examples) directory for more details and [Mol2vec notebooks](https://github.com/samoturk/mol2vec_notebooks) 
repository for visualisations made to easily run in Binder. [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/samoturk/mol2vec_notebooks/master?filepath=Notebooks%2FExploring_Mol2vec_vectors.ipynb)

### Command line tool
Mol2vec is an unsupervised machine learning approach to learn vector representations of molecular substructures.
Command line application has subcommands to prepare a corpus from molecular data (SDF or SMILES), train Mol2vec model
and featurize new samples.

#### Subcommand 'corpus'

Generates corpus to train Mol2vec model. It generates morgan identifiers (up to selected radius) which represent words (molecules are sentences). Words are ordered in the sentence according to atom order in canonical SMILES (generated when generating corpus) and at each atom starting by identifier at radius 0.  
    Corpus subcommand also optionally replaces rare identifiers with selected string (e.g. UNK) which can be later used to represent completely new substructures (i.e. at featurization step). NOTE: It saves the corpus with replaced uncommon identifiers in separate file with ending "_{selected string to replace uncommon}". Since this is unsupervised method we recommend using as much molecules as possible (e.g. complete ZINC database).

##### Performance:  
Corpus generation using 20M compounds with replacement of uncommon identifiers takes 6 hours on 4 cores.  

##### Example:  
To prepare a corpus using radius 1, 4 cores, replace uncommon identifiers that appear <= 3 times with 'UNK' run:
        `mol2vec corpus -i mols.smi -o mols.cp -r 1 -j 4 --uncommon UNK --threshold 3`
          

#### Subcommand 'train'

Trains Mol2vec model using previously prepared corpus.
    
##### Performance:
Training the model on 20M sentences takes ~2 hours on 4 cores.
    
##### Example:
To train a Mol2vec model on corpus with replaced uncommon identifiers using Skip-gram, window size 10, generating 300 dimensional vectors and using 4 cores run:
        `mol2vec train -i mols.cp_UNK -o model.pkl -d 300 -w 10 -m skip-gram --threshold 3 -j 4`
    
    
#### Subcommand 'featurize'

Featurizes new samples using pre-trained Mol2vec model. It saves the result in CSV file with columns for molecule identifiers, canonical SMILES (generated during featurization) and all potential SD fields from input SDF file and finally followed by mol2vec-{0 to n-1} where n is dimensionality of embeddings in the model.  
    
##### Example:
To featurize new samples using pre-trained embeddings and using vector trained on uncommon samples to represent new substructures:
        `mol2vec featurize -i new.smi -o new.csv -m model.pkl -r 1 --uncommon UNK`


For more detail on individual subcommand run:
    `mol2vec $sub-command --help`

### How to cite?
```bib
@article{doi:10.1021/acs.jcim.7b00616,
author = {Jaeger, Sabrina and Fulle, Simone and Turk, Samo},
title = {Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition},
journal = {Journal of Chemical Information and Modeling},
volume = {0},
number = {ja},
pages = {null},
year = {0},
doi = {10.1021/acs.jcim.7b00616},

URL = {http://dx.doi.org/10.1021/acs.jcim.7b00616},
eprint = {http://dx.doi.org/10.1021/acs.jcim.7b00616}
}
```

### Sponsor info
Initial development was supported by [BioMed X Innovation Center](https://bio.mx), Heidelberg.
