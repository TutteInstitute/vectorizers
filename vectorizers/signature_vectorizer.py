import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#import iisignature

NUMPY_SHAPE_ERROR_MSG = """
Error: SignatureVectorizer expects numpy arrays to be of shape (num_samples x path_len x path_dim).
"""
LIST_SHAPE_ERROR_MSG = """
Error: Expecting list entries to be numpy arrays of shape (path_len x path_dim).
"""


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
        If True returns the log-signature (a compressed version of the path signature.
        Otherwise returns the path signature.

    basepoint: bool (default=False)
        If True, prepends each path with the zero vector. The default path signature is blind
        to translational shifts in the paths; use this flag if you care about path translations.
    """

    def __init__(
        self, truncation_level: int = 2, log: bool = False, basepoint: bool = False
    ):
        try:
            global iisignature
            import iisignature as ii
            iisignature = ii
        except ImportError as err:
            from textwrap import dedent
            err.msg += dedent(
                """

                A small bug with the install script of the iisignature makes it
                impossible to install into an environment where its Numpy dependency
                has not yet been installed. Thus, the Vectorizers library does not
                make it an explicit dependency. However, you may install this package
                yourself into this environment now, by running the command

                pip install iisignature

                The problem has been reported to the maintainers of iisignature, and
                this inconvenience will disappear in future releases.
                """
            )
            raise

        assert (
            type(truncation_level) is int
        ), "Error: expecting int type for truncation_level."
        assert type(log) is bool, "Error: expecting bool type for log."
        assert type(basepoint) is bool, "Error: expecting bool type for basepoint"

        self.truncation_level = truncation_level
        self.log = log
        self.basepoint = basepoint

    def fit(self, X, y=None, **fit_params):
        """
        Parameters
        ----------
        X: np.array of shape (n_samples, path_len, path_dim) or list of np.arrays of shape (?, path_dim)
            The path data on which we fit the vectorizer.
            If paths are all the same length, then we can pass them to fit as a numpy array (n_samples, path_len, path_dim).
            If paths are varting length, then we can pass a list of length n_samples, where each entry is a numpy array
            with shape (path_len_i, path_dim). The path_dim should be consistent across the list, but the path length
            can vary/
        """
        if type(X) is np.ndarray:
            assert len(X.shape) == 3, NUMPY_SHAPE_ERROR_MSG
            # We have an array N x p x d of paths
            # all paths have the same length -> batch vectorize
            self.in_dim_ = X.shape[2]
        else:
            assert type(X) is list, "Error: Expecting numpy array or list of paths."
            assert (
                type(X[0]) is np.ndarray
            ), "Error: Expecting list entries to be numpy arrays."
            assert (
                type(X[0]) is np.ndarray and len(X[0].shape) == 2
            ), LIST_SHAPE_ERROR_MSG
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
        """
        Parameters
        ----------
        X: np.array of shape (n_samples, path_len, path_dim) or list of np.arrays of shape (?, path_dim)
            The path data on which we fit the vectorizer.
            If paths are all the same length, then we can pass them to fit as a numpy array (n_samples, path_len, path_dim).
            If paths are varting length, then we can pass a list of length n_samples, where each entry is a numpy array
            with shape (path_len_i, path_dim). The path_dim should be consistent across the list, but the path length
            can vary.

        Returns
        -------
        sigs: np.array of shape (n_samples, self.out_dim_)
            The array of signatures corresponding to the paths given in X, truncated at the truncation level specified
            at initialisation.

        """
        check_is_fitted(
            self,
            [
                "in_dim_",
                "out_dim_",
                "s_",
            ],
        )

        if type(X) is np.ndarray:
            assert len(X.shape) == 3, NUMPY_SHAPE_ERROR_MSG
            # We have an array N x p x d of paths
            # all paths have the same length -> batch vectorize
            assert (
                X.shape[2] == self.in_dim_
            ), "Error: Expecting path_dim to be %d, got path_dim %d." % (
                self.in_dim_,
                X.shape[2],
            )
            if self.basepoint:
                basepoint = np.zeros((X.shape[0], 1, X.shape[2]))
                X = np.concatenate([basepoint, np.array(X)], axis=1)

            if self.log:
                v = iisignature.logsig(X, self.s_)
            else:
                v = iisignature.sig(X, self.truncation_level)
        else:
            # Accepts a list of paths with differing lengths
            assert type(X) is list, "Error: Expecting numpy array or list of paths."
            assert (
                type(X[0]) is np.ndarray
            ), "Error: Expecting list entries to be numpy arrays."
            assert len(X[0].shape) == 2, LIST_SHAPE_ERROR_MSG
            assert (
                X[0].shape[1] == self.in_dim_
            ), "Error: Expecting path_dim to be %d, got path_dim %d." % (
                self.in_dim_,
                X[0].shape[1],
            )
            N = len(X)
            if self.basepoint:
                basepoint = np.zeros(shape=(1, self.in_dim_))
                X = [np.concatenate([basepoint, x], axis=0) for x in X]

            if self.log:
                sig_vectorizer = lambda path: iisignature.logsig(path, self.s_)
            else:
                sig_vectorizer = lambda path: iisignature.sig(
                    path, self.truncation_level
                )

            v = np.empty(shape=(N, self.out_dim_))

            for i, path in enumerate(X):
                assert (
                    path.shape[-1] == self.in_dim_
                ), "Error: Not all paths share the same dimension."
                v[i] = sig_vectorizer(path)

        return v

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
