import numpy as np
import pandas as pd
import pomegranate as pm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
)
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def distribution_type_from_series(series): # pragma: no cover
    if series.dtype in (np.int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        if series.min() >= 0:
            if series.max() == 1:
                return pm.BernoulliDistribution
            else:
                return pm.PoissonDistribution
        elif series.unique().shape[0] <= 50:
            return pm.DiscreteDistribution
        else:
            return pm.NormalDistribution

    elif series.dtype in (np.float, np.float16, np.float32, np.float64):
        if series.min() >= 0:
            if series.max() <= 1:
                return pm.BetaDistribution
            else:
                return pm.GammaDistribution
        else:
            return pm.NormalDistribution

    elif series.dtype in pd.CategoricalDtype:
        return pm.DiscreteDistribution

    else:
        raise ValueError(f"Failed to handle series {series}")


def preprocess_dataframe(df, time_granularity="1s"): # pragma: no cover
    for feature in df:
        if feature.dtype == object:
            df[feature] = pd.Categorical(df[feature])
        elif is_datetime(df[feature]):
            df[feature] = ((df.feature - df[feature].min()) / pd.Timedelta(time_granularity))

    return

class DataframeDistributionVectorizer(BaseEstimator, TransformerMixin): # pragma: no cover

    def __init__(self, n_components=100):
        self.n_components = n_components

    def fit(self, X, y=None, **fit_params):
        if type(X) == pd.DataFrame:
            X = preprocess_dataframe(X.copy())
            column_models = [
                distribution_type_from_series(X[feature])
                for feature in X
            ]
        elif type(X) == np.ndarray:
            column_models = [
                distribution_type_from_series(X[:, i])
                for i in range(X.shape[1])
            ]
        else:
            raise ValueError(f"Input type {type(X)} is not currently supported")

        self.mixture_model_ = pm.GeneralMixtureModel.from_samples(
            column_models, n_components=self.n_components, X=X
        )

    def transform(self, X):
        check_is_fitted(self, ["mixture_model_"])

        if type(X) == pd.DataFrame:
            X = preprocess_dataframe(X.copy())

        return self.mixture_model_.predict_proba(X)
