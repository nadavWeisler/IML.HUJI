import numpy as np
import pandas as pd
import plotly.io as pio
from matplotlib import pyplot as plt

from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    temp_df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    temp_df = temp_df[(temp_df.Year > 0) & (temp_df.Month > 0) & (temp_df.Day > 0)]
    temp_df['Dayofyear'] = temp_df['Date'].dt.dayofyear
    return temp_df[temp_df["Temp"] > -60]


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    temp_df = load_data("../datasets/city_temperature.csv")

    # Question 2 - Exploring data for specific country
    temp_df[temp_df["Country"] == "Israel"].plot(
        kind='scatter', x='Dayofyear', y='Temp', c="Year", colormap='Paired')
    temp_df[temp_df["Country"] == "Israel"].groupby('Month').Temp.mean().plot(
        kind="bar", yerr=temp_df[temp_df["Country"] == "Israel"].groupby('Month').Temp.std())

    # Question 3 - Exploring differences between countries
    for country in temp_df["Country"].unique():
        country_df = temp_df[temp_df["Country"] == country]
        month_std = country_df.groupby("Month").agg('std')
        month_avg = country_df.groupby("Month").agg('mean')
        plt.errorbar(x=np.arange(1, 13), y=month_avg["Temp"], yerr=month_std["Temp"], label=country)
    plt.legend()
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    X_train, y_train, X_test, y_test = split_train_test(
        pd.DataFrame(temp_df[temp_df["Country"] == "Israel"]["Dayofyear"]),
        temp_df[temp_df["Country"] == "Israel"]["Temp"])

    losses_lst = []
    for k in range(1, 11):
        polynomialRegressor = PolynomialFitting(k)
        polynomialRegressor.fit(X_train.to_numpy().flatten(), y_train.to_numpy().flatten())
        loss = polynomialRegressor.loss(X_test.to_numpy().flatten(), y_test.to_numpy().flatten())
        losses_lst.append(loss)
        print("Loss for k=", k, ":", np.round(loss, 2))
    plt.bar(np.arange(1, 11), losses_lst)
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    il_model = PolynomialFitting(np.argwhere(np.min(losses_lst) == losses_lst)[0][0] + 1)
    il_model.fit(X=temp_df[temp_df["Country"] == "Israel"]["Dayofyear"],
                 y=temp_df[temp_df["Country"] == "Israel"]["Temp"])
    countries = temp_df[temp_df.Country != "Israel"].Country.unique()

    countries_losses = []
    for country in countries:
        countries_losses.append(il_model.loss(temp_df[temp_df["Country"] == country]["Dayofyear"],
                                              temp_df[temp_df["Country"] == country]["Temp"]))
    plt.bar(x=countries, height=countries_losses)
    plt.show()
