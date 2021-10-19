.. -*- mode: rst -*-

.. image:: doc/vectorizers_logo_text.png
  :width: 600
  :alt: Vectorizers Logo

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.com/TutteInstitute/vectorizers.svg?branch=master
.. _Travis: https://travis-ci.com/TutteInstitute/vectorizers

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/sjawsgwo7g4k3jon?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/lmcinnes/vectorizers

.. |Codecov| image:: https://codecov.io/gh/TutteInstitute/vectorizers/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/TutteInstitute/vectorizers


.. |CircleCI| image:: https://circleci.com/gh/TutteInstitute/vectorizers.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/vectorizers/badge/?version=latest
.. _ReadTheDocs: https://vectorizers.readthedocs.io/en/latest/?badge=latest

===========
Vectorizers
===========

There are a large number of machine learning tools for effectively exploring and working
with data that is given as vectors (ideally with a defined notion of distance as well).
There is also a large volume of data that does not come neatly packaged as vectors. It
could be text data, variable length sequence data (either numeric or categorical),
dataframes of mixed data types, sets of point clouds, or more. Usually, one way or another,
such data can be wrangled into vectors in a way that preserves some relevant properties
of the original data. This library seeks to provide a suite of a wide variety of
general purpose techniques for such wrangling, making it easier and faster for users
to get various kinds of unstructured sequence data into vector formats for exploration and
machine learning.

--------------------
Why use Vectorizers?
--------------------

Data wrangling can be tedious, error-prone, and fragile when trying to integrate it into
production pipelines. The vectorizers library aims to provide a set of easy to use
tools for turning various kinds of unstructured sequence data into vectors. By following the
scikit-learn transformer API we ensure that any of the vectorizer classes can be
trivially integrated into existing sklearn workflows or pipelines. By keeping the
vectorization approaches as general as possible (as opposed to specialising on very
specific data types), we aim to ensure that a very broad range of data can be handled
efficiently. Finally we aim to provide robust techniques with sound mathematical foundations
over potentially more powerful but black-box approaches for greater transparency
in data processing and transformation.

----------------------
How to use Vectorizers
----------------------

Quick start examples to be added soon ...

For further examples on using this library for text we recommend checking out the documentation
written up in the EasyData reproducible data science framework by some of our colleagues over at:
https://github.com/hackalog/vectorizers_playground

----------
Installing
----------

Vectorizers is designed to be easy to install being a pure python module with
relatively light requirements:

* numpy
* scipy
* scikit-learn >= 0.22
* numba >= 0.51

In the near future the package should be pip installable -- check back for updates:

.. code:: bash

    pip install vectorizers

To manually install this package:

.. code:: bash

    wget https://github.com/TutteInstitute/vectorizers/archive/master.zip
    unzip master.zip
    rm master.zip
    cd vectorizers-master
    python setup.py install

----------------
Help and Support
----------------

This project is still young. The `documentation <https://vectorizers.readthedocs.io/en/latest/>`_ is still growing. In the meantime please
`open an issue <https://github.com/TutteInstitute/vectorizers/issues/new>`_
and we will try to provide any help and guidance that we can. Please also check
the docstrings on the code, which provide some descriptions of the parameters.

------------
Contributing
------------

Contributions are more than welcome! There are lots of opportunities
for potential projects, so please get in touch if you would like to
help out. Everything from code to notebooks to
examples and documentation are all *equally valuable* so please don't feel
you can't contribute. We would greatly appreciate the contribution of
tutorial notebooks applying vectorizer tools to diverse or interesting
datasets. If you find vectorizers useful for your data please consider
contributing an example showing how it can apply to the kind of data
you work with!


To contribute please `fork the project <https://github.com/TutteInstitute/vectorizers/issues#fork-destination-box>`_ make your changes and
submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.

-------
License
-------

The vectorizers package is 2-clause BSD licensed.


