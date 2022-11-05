import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:
    import iisignature

    HAVE_IISIGNATURE = True
except ModuleNotFoundError:
    HAVE_IISIGNATURE = False


class SignatureVectorizer(BaseEstimator, TransformerMixin):
    """Transforms a list or array of paths into their path signatures.

    Uses the iisignature library (https://pypi.org/project/iisignature/)
        * pip install iisignature

    For more details on the path signature technique, please refer to:
        * Rough paths, Signatures and the modelling of functions on streams. Lyons, T. (2014)
          https://arxiv.org/pdf/1405.4537.pdf
        * A Primer on the Signature Method in Machine Learning. Cheyrev, I. (2016)
          https://arxiv.org/pdf/1603.03788.pdf

    Parameters
    ----------
    truncation_level: int (default = 2)
        The level at which we truncate the infinite signature.

    log: bool (default=False)
        If True returns the logsignature. Otherwise returns the signature.
    """

    def __init__(self, truncation_level=2, log=False):

        if not HAVE_IISIGNATURE:
            raise Exception("SignatureVectorizer requires iisignature package.")

        self.truncation_level = truncation_level
        self.log = log

    def fit(self, X, y=None, **fit_params):
        if type(X) is np.array and len(X.shape) == 3:
            # We have an array N x p x d of paths
            # all paths have the same length -> batch vectorize
            self.in_dim_ = X.shape[2]
        else:
            # Accepts a list of paths with differing lengths
            self.in_dim_ = X[0].shape[1]

        if self.log:
            self.s_ = iisignature.prepare(self.in_dim_, self.truncation_level)
            self.out_dim_ = iisignature.logsiglength(
                self.in_dim_, self.truncation_level
            )
        else:
            self.s_ = None
            self.out_dim_ = iisignature.siglength(self.in_dim_, self.truncation_level)

    def transform(self, X):

        check_is_fitted(
            self,
            [
                "in_dim_",
                "out_dim_",
                "s_",
            ],
        )

        if type(X) is np.array and len(X.shape) == 3:
            # We have an array N x p x d of paths
            # all paths have the same length -> batch vectorize
            if self.log:
                v = iisignature.logsig(X, self.s_)
            else:
                v = iisignature.sig(X, self.truncation_level)
        else:
            # Accepts a list of paths with differing lengths
            N = len(X)
            if self.log:
                sig_vectorizer = lambda path: iisignature.logsig(path, self.s_)
            else:
                sig_vectorizer = lambda path: iisignature.sig(
                    path, self.truncation_level
                )

            v = np.empty(shape=(N, self.out_dim_))

            for i, path in enumerate(X):
                assert path.shape[-1] == self.in_dim_
                v[i] = sig_vectorizer(path)

        return v

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
