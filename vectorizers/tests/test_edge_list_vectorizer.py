import pytest

from vectorizers import EdgeListVectorizer

# from vectorizers.edge_list_vectorizer import read_edge_data
import numpy as np
import pandas as pd

rows = np.array(["a", "b", "c", "d", "d"])
cols = np.array(["b", "c", "d", "b", "c"])
vals = np.array([1, 2, 3, 4, 8])
test_data = (rows, cols, vals)
list_of_edges = [
    ["a", "b", 1],
    ["b", "c", 2],
    ["c", "d", 3],
    ["d", "b", 4],
    ["d", "c", 8],
]
df_of_edges = pd.DataFrame({"r": rows, "c": cols, "v": vals})


# Tuple or list of columns, data frame, list of edges
@pytest.mark.parametrize(
    "data", [(rows, cols, vals), [rows, cols, vals], list_of_edges, df_of_edges]
)
def test_edgelist_input(data):
    model = EdgeListVectorizer().fit(data)
    result = model.transform(data)
    result1 = EdgeListVectorizer().fit_transform(data)
    assert np.allclose(result.toarray(), result1.toarray())
    assert result.shape == (4, 3)
    assert np.allclose(result.toarray()[:, 1], np.array([0, 2, 0, 8]))


def test_edgelist_specified_rows():
    row_dict = {"a": 0, "d": 1}
    result = EdgeListVectorizer(row_label_dictionary=row_dict).fit_transform(test_data)
    assert result.shape == (2, 3)
    assert np.allclose(result.toarray()[1, :], np.array([4, 8, 0]))


def test_edgelist_specified_columns():
    column_dict = {"b": 0, "c": 1}
    result = EdgeListVectorizer(column_label_dictionary=column_dict).fit_transform(
        test_data
    )
    assert result.shape == (4, 2)
    assert np.allclose(result.toarray()[:, 1], np.array([0, 2, 0, 8]))


def test_edgelist_specified_rows_missing_index():
    row_dict = {"a": 2, "d": 4}
    result = EdgeListVectorizer(row_label_dictionary=row_dict).fit_transform(test_data)
    assert result.shape == (5, 3)
    assert np.allclose(result.toarray()[:, 0], np.array([0, 0, 1, 0, 4]))


def test_edgelist_specified_column_missing_index():
    column_dict = {"b": 2, "c": 4}
    result = EdgeListVectorizer(column_label_dictionary=column_dict).fit_transform(
        test_data
    )
    assert result.shape == (4, 5)
    assert np.allclose(result.toarray()[:, 4], np.array([0, 2, 0, 8]))


# TODO: Write a unit test for joint_space=True
