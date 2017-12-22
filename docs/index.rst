.. highlight:: python

*********************
Mol2vec documentation
*********************
`Mol2vec <http://pubs.acs.org/doi/10.1021/acs.jcim.7b00616>`_
- an unsupervised machine learning approach to learn vector representations of molecular substructures

.. contents::
    :depth: 3

Installation
============

.. code-block:: bash

    pip install git+https://github.com/samoturk/mol2vec

.. note:: RDKit has to be installed manually and is not automatically installed by pip as a dependency.

Requirements
````````````
* Python 3 (Python 2.x is `not supported <http://www.python3statement.org/>`_)
* `NumPy <http://www.numpy.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `seaborn <https://seaborn.pydata.org/>`_
* `pandas <http://pandas.pydata.org/>`_
* `IPython <https://ipython.org/>`_
* `RDKit <http://www.rdkit.org/docs/Install.html>`_
* `scikit-learn <http://scikit-learn.org/stable/>`_
* `gensim <https://radimrehurek.com/gensim/>`_
* `tqdm <https://pypi.python.org/pypi/tqdm>`_
* `joblib <https://pythonhosted.org/joblib/>`_

Building the documentation
``````````````````````````
To build the documentation install `sphinx`, `numpydoc` and `sphinx_rtd_theme` and then run in the docs directory:

.. code-block:: bash

    make html

Usage
=====

As python module
````````````````
.. code-block:: python

    from mol2vec import features
    from mol2vec import helpers

First line imports functions to generate “sentences” from molecules and train the model, and second line imports
functions useful for depictions. Check `examples <https://github.com/samoturk/mol2vec/examples>`_ directory for more
details and `Mol2vec notebooks <https://github.com/samoturk/mol2vec_notebooks>`_ repository for visualisations made to
easily run in Binder.

Command line application
````````````````````````
Command line application has subcommands to prepare a corpus from molecular data (SDF or SMILES), train Mol2vec model
and featurize new samples.
To get help from Mol2vec command line application:

.. code-block:: bash

    mol2vec --help

For more detail on individual subcommands run:

.. code-block:: bash

    mol2vec $sub-command --help



Subcommand 'corpus'
+++++++++++++++++++

Generates corpus to train Mol2vec model. It generates morgan identifiers (up to selected radius) which represent words
(molecules are sentences). Words are ordered in the sentence according to atom order in canonical SMILES (generated when
generating corpus) and at each atom starting by identifier at radius 0.
Corpus subcommand also optionally replaces rare identifiers with selected string (e.g. UNK) which can be later used to
represent completely new substructures (i.e. at featurization step). NOTE: It saves the corpus with replaced uncommon
identifiers in separate file with ending "_{selected string to replace uncommon}". Since this is unsupervised method we
recommend using as much molecules as possible (e.g. complete ZINC database).

.. note:: Corpus generation using 20M compounds with replacement of uncommon identifiers takes 6 hours on 4 cores.

To prepare a corpus using radius 1, 4 cores, replace uncommon identifiers that appear <= 3 times with 'UNK' run:

.. code-block:: bash

    mol2vec corpus -i mols.smi -o mols.cp -r 1 -j 4 --uncommon UNK --threshold 3


Subcommand 'train'
++++++++++++++++++

Trains Mol2vec model using previously prepared corpus.

.. note:: Training the model on 20M sentences takes ~2 hours on 4 cores.

To train a Mol2vec model on corpus with replaced uncommon identifiers using Skip-gram, window size 10, generating 300
dimensional vectors and using 4 cores run:

.. code-block:: bash

        mol2vec train -i mols.cp_UNK -o model.pkl -d 300 -w 10 -m skip-gram --threshold 3 -j 4


Subcommand 'featurize'
++++++++++++++++++++++

Featurizes new samples using pre-trained Mol2vec model. It saves the result in CSV file with columns for molecule
identifiers, canonical SMILES (generated during featurization) and all potential SD fields from input SDF file and
finally followed by mol2vec-{0 to n-1} where n is dimensionality of embeddings in the model.

To featurize new samples using pre-trained embeddings and using vector trained on uncommon samples to represent new
substructures:

.. code-block:: bash

        mol2vec featurize -i new.smi -o new.csv -m model.pkl -r 1 --uncommon UNK


How to cite?
============

.. code-block:: bib

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

.. toctree::
   :maxdepth: 1
   :caption: Contents:

API documentation
=================

.. automodule:: mol2vec.features
   :members:

.. automodule:: mol2vec.helpers
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
