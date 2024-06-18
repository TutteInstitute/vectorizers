###############
Vectorizers API
###############

Ngram and Skipgram Vectorizer
===============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   NgramVectorizer
.. autoclass:: vectorizers.NgramVectorizer
   SkipgramVectorizer
.. autoclass:: vectorizers.SkimgramVectorizer
   LZCompressionVectorizer
   BytePairEncodingVectorizer

TokenCooccurrenceVectorizers
===========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   TokenCooccurrenceVectorizer
   MultiSetCooccurrenceVectorizer
   TimedTokenCooccurrenceVectorizer
   LabelledTreeCooccurrenceVectorizer

Wasserstein style Vectorizers
=============================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   WassersteinVectorizer
   SinkhornVectorizer
   ApproximateWassersteinVectorizer

Utility Vectorizers and Transformers
====================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   EdgeListVectorizer
   CategoricalColumnTransformer
   InformationWeightTransformer
.. autoclass:: vectorizers.transformers.InformationWeightTransformer
   RowDenoisingTransformer
   CountFeatureCompressionTransformer

Time Series Vectorizers and Transformers
========================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HistogramVectorizer
   KDEVectorizer
   SlidingWindowTransformer




