###############
Vectorizers API
###############

.. currentmodule:: vectorizers

TokenCooccurrenceVectorizer
===========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   TokenCooccurrenceVectorizer

NgramVectorizer
===============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   NgramVectorizer

SkipgramVectorizer
==================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SkipgramVectorizer

WassersteinVectorizer
=====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   WassersteinVectorizer

SinkhornVectorizer
==================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SinkhornVectorizer

ApproximateWassersteinVectorizer
================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ApproximateWassersteinVectorizer

DistributionVectorizer
======================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   DistributionVectorizer

HistogramVectorizer
===================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HistogramVectorizer

KDEVectorizer
=============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   KDEVectorizer

.. currentmodule:: vectorizers.transformers

The ``transformers`` submodule provides a number of utility
transformers that, rather than directly providing vectorization,
transform data to make vectorization easier with the standard
vectorization tools provided, or transform vectorized data based
on assumptions about the kinds of output vectorizers often produce.

CategoricalColumnTransformer
============================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   CategoricalColumnTransformer

InformationWeightTransformer
============================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   InformationWeightTransformer

RowDenoisingTransformer
========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   RowDenoisingTransformer

CountFeatureCompressionTransformer
==================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   CountFeatureCompressionTransformer

SlidingWindowTransformer
========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SlidingWindowTransformer


