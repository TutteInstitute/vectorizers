############################
Quick Start with Vectorizers
############################

Vectorizers provides a number of tools for working with various kinds of
unstructured data with a focus on sequence data. The library is built to be
compatible with scikit-learn_ and can be used in scikit-learn pipelines.

----------
Installing
----------

Vectorizers can be installed via pip (coming soon) and via conda-forge (coming later).

(Coming soon)
.. code:: bash

    pip install vectorizers

(Currently available)
.. code:: bash

    pip install git+https://github.com/TutteInstitute/vectorizers.git

To manually install this package:

.. code:: bash

    wget https://github.com/TutteInstitute/vectorizers/archive/master.zip
    unzip master.zip
    rm master.zip
    cd vectorizers-master
    python setup.py install

-----------
Basic Usage
-----------

The vetcorizers package provides a number of tools for vectorizing different kinds of
input data. All of them are available as classes that follow sciki-learn's basic API
for transformers, converting input data into vectors in one form or another. For example
to convert sequences of categorical data into ngram vector representations one might use

.. code:: python3

    import vectorizers

    ngrammer = vectorizers.NgramVectorizer(ngram_size=2)
    ngram_vetcors = ngrammer.fit_transform(input_sequences)

These classes can easily be fit into sklearn pipelines, passing vector
representations on to other scikit-learn (or scikit-learn compatible) classes. See
the `Vectorizers API`_ documentation for more details on the available classes.

Vetcorizers also provides a number of utility transformers in the ``vectorizers.transformers``
namespace. These provide convenience transformations of data -- either transforms on vectorized
data, including feature weighting tools, or transformations of structured and unstructured data
into sequences more amenable to other vectorizers classes.

.. _scikit-learn: https://scikit-learn.org/stable/
