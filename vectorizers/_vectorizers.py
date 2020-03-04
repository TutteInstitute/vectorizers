"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TokenCooccurenceVectorizer(BaseEstimator, TransformerMixin):

    pass


class DistributionVectorizer(BaseEstimator, TransformerMixin):

    pass


class HistogramVectorizer(BaseEstimator, TransformerMixin):

    pass


class SkipgramVectorizer(BaseEstimator, TransformerMixin):

    pass


class NgramVectorizer(BaseEstimator, TransformerMixin):

    pass


class Wasserstein1DHistogramTransformer(BaseEstimator, TransformerMixin):

    pass