import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from warnings import warn


class CategoricalColumnTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer is useful for describing an object as a bag of the categorical values that
    have been used to represent it within a pandas DataFrame.

    It takes an categorical column name to groupby, object_column_name, and one
    or more categorical columns to be used to describe these objects,
    descriptor_column_name.  Then it returns a Series with an index being the
    unique entries of your object_column_name and the values being a list of
    the appropriate categorical values from your descriptor_column_name.

    It can be thought of as a PivotTableTransformer if you'd like.

    Parameters
    ----------
    object_column_name: string
        The column name from the DataFrame where our object values can be found.
        This will be the thing we are grouping by.

    descriptor_column_name: string or list
        The name or names of the categorical column(s) who's values will be used for describing our
        objects.  If you are using multiple names it's recommended that you set include_column_name=True.

    include_column_name: bool (default = False)
        Should the column name be appended at the beginning of each value?
        This is useful if you intend to combine values from multiple categorical columns
        after the fact.

    unique_values: bool (default = False)
        Should we apply a unique to the values in column before building our list representation?

    """

    def __init__(
        self,
        object_column_name,
        descriptor_column_name,
        include_column_name=False,
        unique_values=False,
    ):
        self.object_column_name = object_column_name
        self.descriptor_column_name = descriptor_column_name
        # Get everything on consistent footing so we don't have to handle multiple cases.
        if type(self.descriptor_column_name) == str:
            self.descriptor_column_name_ = [self.descriptor_column_name]
        else:
            self.descriptor_column_name_ = self.descriptor_column_name
        self.include_column_name = include_column_name
        self.unique_values = unique_values

        if (
            (self.include_column_name is False)
            and (type(self.descriptor_column_name) == list)
            and (len(self.descriptor_column_name) > 1)
        ):
            warn(
                "It is recommended that if you are aggregating "
                "multiple columns that you set include_column_name=True"
            )

    def fit_transform(self, X, y=None, **fit_params):
        """
        This transformer is useful for describing an object as a bag of the categorical values that
        have been used to represent it within a pandas DataFrame.

        It takes an categorical column name to groupby, object_column_name, and one or more
        categorical columns to be used to describe these objects, descriptor_column_name.
        Then it returns a Series with an index being the unique entries of your object_column_name
        and the values being a list of the appropriate categorical values from your descriptor_column_name.

        Parameters
        ----------
        X: pd.DataFrame
            a pandas dataframe with columns who's names match those specified in the object_column_name and
            descriptor_column_name of the constructor.

        Returns
        -------
        pandas Series
            Series with an index being the unique entries of your object_column_name
            and the values being a list of the appropriate categorical values from your descriptor_column_name.
        """
        # Check that the dataframe has the appropriate columns
        required_columns = set([self.object_column_name] + self.descriptor_column_name_)
        if not required_columns.issubset(X.columns):
            raise ValueError(
                f"Sorry the required column(s) {set(required_columns).difference(set(X.columns))} are not "
                f"present in your data frame. \n"
                f"Please either specify a new instance or apply to a different data frame. "
            )

        # Compute a single groupby ahead of time to save on compute
        grouped_frame = X.groupby(self.object_column_name)
        aggregated_columns = []
        for column in self.descriptor_column_name_:
            if self.include_column_name:
                if self.unique_values:
                    aggregated_columns.append(
                        grouped_frame[column].agg(
                            lambda x: [
                                column + ":" + value
                                for value in x.unique()
                                if pd.notna(value)
                            ]
                        )
                    )
                else:
                    aggregated_columns.append(
                        grouped_frame[column].agg(
                            lambda x: [
                                column + ":" + value for value in x if pd.notna(value)
                            ]
                        )
                    )
            else:
                if self.unique_values:
                    aggregated_columns.append(
                        grouped_frame[column].agg(
                            lambda x: [value for value in x.unique() if pd.notna(value)]
                        )
                    )
                else:
                    aggregated_columns.append(
                        grouped_frame[column].agg(
                            lambda x: [value for value in x if pd.notna(value)]
                        )
                    )
        reduced = pd.concat(aggregated_columns, axis="columns").sum(axis=1)
        return reduced

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self
