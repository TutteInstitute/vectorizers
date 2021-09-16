.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Vectorizers: Transform unstructured data into vectors
=====================================================

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

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Quick Start

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started Tutorials

   user_guide

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: In Depth Tutorials

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Example Use Cases

   TokenCooccurrenceVectorizer_multi_labelled_cyberEvents
   CategoricalColumnTransformer_documentation

.. toctree::
   :caption: API Reference:

   api

`Getting started <quick_start.html>`_
-------------------------------------

Information regarding this template and how to modify it for your own project.

`User Guide <user_guide.html>`_
-------------------------------

An example of narrative documentation.

`API Documentation <api.html>`_
-------------------------------

An example of API documentation.

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples. It complements the `User Guide <user_guide.html>`_.