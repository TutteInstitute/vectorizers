import numpy as np
import numba

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import scipy.sparse


def read_edge_data(X):
    """
    Read in data of various forms and converts them into an np.array of [row_labels, column_labels, values]

    Returns
    -------
    N x 3 np.array of [row_labels, column_labels, values]
    """
    try:
        edge_list = np.array(X, dtype=object)
    except:
        raise ValueError("Couldn't convert for your data format into an numpy array.")
    if (edge_list.shape[1] != 3) & (edge_list.shape[0] == 3):
        edge_list = edge_list.T
    if edge_list.shape[1] != 3:
        raise ValueError(
            f"Incorrect format of data passed in.  "
            f"We expected some format of Nx3 data and received {edge_list.shape[0]} by {edge_list.shape[1]} data"
        )

    # TODO: Test if edge_list[:,2] is numeric. We currently just convert it into a float.  I'd rather preserve the type.
    return edge_list


class EdgeListVectorizer(BaseEstimator, TransformerMixin):
    """
    Takes a weighted edge list of the form row_labels, column_labels, value
    and represents each row_name as a sparse matrix containing the values
    associated with each column_name.

    This might also be thought of as a PivotTableVectorizer or a CrossTabVectorizer.

    Parameters
    ----------
    column_label_dictionary: dictionary or None (optional, default=None)
        A fixed dictionary mapping tokens to indices, or None if the dictionary
        should be learned from the training data.  If specified this will limit
        the tokens
    row_label_dictionary: dictionary or None (optional, default=None)
        A fixed dictionary mapping row labels to indices, or None if the dictionary
        should be learned from the training data.  If specified this will limit
        the tokens
    joint_space: Bool (optional, default=False)
        Are the first two columns of your edge list over the same token space.  If so build
        a single unified token dictionary over both columns.
    pre_indexed: bool (optional, default=False)
        Not yet implemented.  I'm not sure that this feature is going to be used enought to prioritize implementing.
        Please reach out if this would be useful to you.
        My row and columns are just the row and column indices and not row and column labels.

    """

    def __init__(
        self,
        column_label_dictionary=None,
        row_label_dictionary=None,
        joint_space=False,
    ):
        self.column_label_dictionary = column_label_dictionary
        self.row_label_dictionary = row_label_dictionary
        self.joint_space = joint_space

    def fit(self, X, y=None, **fit_params):
        # Convert data from whatever format it came in into an Nx3 np.array
        self.edge_list_ = read_edge_data(X)

        if self.joint_space:
            if self.column_label_dictionary is None:
                if self.row_label_dictionary is None:
                    self.row_label_dictionary_ = {
                        token: index
                        for index, token in enumerate(
                            np.unique(
                                np.append(self.edge_list_[:, 0], self.edge_list_[:, 1])
                            )
                        )
                    }
                self.column_label_dictionary_ = self.row_label_dictionary_
            elif self.row_label_dictionary is None:
                self.column_label_dictionary_ = self.column_label_dictionary
                self.row_label_dictionary_ = self.column_label_dictionary
            elif self.column_label_dictionary is None:
                self.column_label_dictionary_ = self.row_label_dictionary
                self.row_label_dictionary_ = self.row_label_dictionary
            else:
                raise ValueError(
                    "Joint_space=True: Please specify at most a single label dictionary (either one works)."
                )
        else:  # Not in a joint space
            if self.row_label_dictionary is None:
                self.row_label_dictionary_ = {
                    token: index
                    for index, token in enumerate(np.unique(self.edge_list_[:, 0]))
                }
            else:
                self.row_label_dictionary_ = self.row_label_dictionary
            if self.column_label_dictionary is None:
                self.column_label_dictionary_ = {
                    token: index
                    for index, token in enumerate(np.unique(self.edge_list_[:, 1]))
                }
            else:
                self.column_label_dictionary_ = self.column_label_dictionary
        # Build reverse indexes
        self.row_index_dictionary_ = {
            y: x for (x, y) in self.row_label_dictionary_.items()
        }
        self.column_index_dictionary_ = {
            y: x for (x, y) in self.column_label_dictionary_.items()
        }
        max_row = np.max(list(self.row_index_dictionary_.keys())) + 1
        max_col = np.max(list(self.column_index_dictionary_.keys())) + 1

        # Get row and column indices for only the edges who have both labels in our dictionary index
        valid_rows = np.isin(
            self.edge_list_[:, 0], list(self.row_label_dictionary_.keys())
        )
        valid_cols = np.isin(
            self.edge_list_[:, 1], list(self.column_label_dictionary_.keys())
        )
        valid_edges = valid_rows & valid_cols
        row_indices = [
            self.row_label_dictionary_[x] for x in self.edge_list_[valid_edges, 0]
        ]
        col_indices = [
            self.column_label_dictionary_[x] for x in self.edge_list_[valid_edges, 1]
        ]
        # Must specify the shape to ensure that tailing zero rows/cols aren't suppressed.
        self._train_matrix = scipy.sparse.coo_matrix(
            (self.edge_list_[valid_edges, 2].astype(float), (row_indices, col_indices)),
            shape=(max_row, max_col),
        ).tocsr()
        self._train_matrix.sum_duplicates()

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self._train_matrix

    def transform(self, X):
        check_is_fitted(
            self,
            [
                "column_label_dictionary_",
                "row_label_dictionary_",
            ],
        )

        edge_list = read_edge_data(X)

        # Get row and column indices for only the edges who have both labels in our dictionary index
        valid_rows = np.isin(edge_list[:, 0], list(self.row_label_dictionary_.keys()))
        valid_cols = np.isin(
            edge_list[:, 1], list(self.column_label_dictionary_.keys())
        )
        valid_edges = valid_rows & valid_cols
        row_indices = [self.row_label_dictionary_[x] for x in edge_list[valid_edges, 0]]
        col_indices = [
            self.column_label_dictionary_[x] for x in edge_list[valid_edges, 1]
        ]

        matrix = scipy.sparse.coo_matrix(
            (edge_list[valid_edges, 2].astype(float), (row_indices, col_indices))
        ).tocsr()
        matrix.sum_duplicates()
        return matrix
