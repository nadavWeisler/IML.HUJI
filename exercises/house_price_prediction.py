from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house_df = pd.read_csv(filename)
    house_df = house_df.drop(labels=["id", "lat", "long"], axis=1).dropna().drop_duplicates()

    house_df.zipcode = house_df.zipcode.astype(int)
    house_df = house_df[
        (house_df.bedrooms > 0) & (house_df.bathrooms > 0) & (house_df.sqft_above > 0) &
        (house_df.sqft_above > 0) & (house_df.floors >= 0) & (house_df.sqft_basement >= 0)
        ]
    house_df["is_renovated_lately"] = np.where(house_df["yr_renovated"] >= 1995, 1, 0)
    house_df = house_df.drop(labels="yr_renovated", axis=1)
    house_df['date'] = pd.to_datetime(house_df['date'])
    house_df['month'] = house_df['date'].dt.month
    house_df['date_day'] = house_df['date'].dt.day
    house_df['weekday'] = house_df['date'].dt.weekday
    house_df['year'] = house_df['date'].dt.year
    house_df['quarter'] = house_df['date'].dt.quarter
    house_df = house_df.drop(labels="date", axis=1)
    house_df["built_decade"] = (house_df["yr_built"] / 10).astype(int)
    house_df = house_df.drop(labels="yr_built", axis=1)
    house_df = pd.get_dummies(house_df, prefix="zipcode", columns=["zipcode"])
    house_df = pd.get_dummies(house_df, prefix="month", columns=["month"])
    house_df = pd.get_dummies(house_df, prefix="built_decade", columns=["built_decade"])
    return house_df.drop(labels="price", axis=1), house_df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    correlation = X.apply(lambda col: y.cov(col) / (y.std() * col.std()))  # pearson's correlation
    for feature, values in X.iteritems():
        plt.scatter(x=values, y=y)
        plt.ylabel('price')
        plt.xlabel(feature)
        plt.title("feature name: " + str(feature) + " correlation with price: " + str(correlation[feature]))
        plt.savefig(output_path + "/" + str(feature) + ".png")
        plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "houses_prices_graphs")

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    losses_mean = []
    variances = []
    for p in np.arange(10, 101):
        current_loss = []
        for i in range(10):
            sampled = X_train.assign(response=y_train).sample(frac=(p / 100))
            ln = LinearRegression()
            ln.fit(sampled.drop(["response"], axis=1).to_numpy(), sampled["response"].to_numpy())
            current_loss.append(ln.loss(X_test.to_numpy(), y_test.to_numpy()))
        current_loss = np.array(current_loss)
        losses_mean.append(np.sum(current_loss) / len(current_loss))
        variances.append(current_loss.var())

    average_losses = np.array(losses_mean)
    variances = np.array(variances)
    plt.scatter(np.arange(10, 101), average_losses)
    plt.errorbar(np.arange(10, 101), average_losses, yerr=np.sqrt(variances) * 2)
    plt.title("Loss of test data by percent of used training data")
    plt.xlabel("Used percent of training data ")
    plt.ylabel("Loss of test data")
    plt.savefig("./losses.png")
    plt.show()
