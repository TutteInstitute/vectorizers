.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: vectorizers_logo_text.png
  :width: 600
  :alt: Vectorizers Logo

=====================================================
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
   :caption: Quick Start

   quick_start

.. toctree::
   :maxdepth: 2
   :caption: Getting Started Tutorials

   document_vectorization
   CategoricalColumnTransformer_intro

.. toctree::
   :maxdepth: 2
   :caption: In Depth Tutorials

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Example Use Cases

   token_cooccurrence_vectorizer_multi_labelled_cyber_example
   categorical_column_transformer_example

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api
