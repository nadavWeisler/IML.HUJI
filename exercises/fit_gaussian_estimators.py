import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(loc=10, scale=1, size=1000)
    universal_gaussian = UnivariateGaussian()
    universal_gaussian.fit(samples)
    print('mu:', universal_gaussian.mu_, 'var:', universal_gaussian.var_)

    # Question 2 - Empirically showing sample mean is consistent
    mu_distances = []
    for sample_size in np.arange(10, 1000, 10):
        universal_gaussian.fit(samples[:sample_size])
        mu_distances.append(abs(universal_gaussian.mu_ - 10))

    fig = px.scatter(x=np.arange(10, 1000, 10), y=mu_distances, title="Sample sizes vs MU distances")
    fig.update_xaxes(title_text="Sample sizes")
    fig.update_yaxes(title_text="MU distances")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_values = universal_gaussian.pdf(samples)
    fig = px.scatter(x=samples, y=pdf_values, title="Samples vs PDF values")
    fig.update_xaxes(title_text="Sample sizes")
    fig.update_yaxes(title_text="PDF values")
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0]).T
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(samples)
    print('mu:', multivariate_gaussian.mu_, 'cov:', multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    space = np.linspace(-10, 10, 200)
    space_pairs = np.transpose(np.array([np.repeat(space, len(space)), np.tile(space, len(space))]))

    def single_func(single_pair):
        return multivariate_gaussian.log_likelihood(np.array([single_pair[0], 0, single_pair[1], 0]),
                                                    multivariate_gaussian.cov_, samples)

    mat = np.array(list(map(single_func, space_pairs)))
    likelihoods = mat.reshape(200, 200)
    fig = go.Figure(go.Heatmap(x=np.linspace(-10, 10, 200), y=np.linspace(-10, 10, 200), z=likelihoods.T),
                    layout=go.Layout(title="Loglikelihood HeatMap"))
    fig.update_xaxes(title_text="f3 values")
    fig.update_yaxes(title_text="f1 values")
    fig.show()

    max_value = np.max(mat)
    f3, f1 = np.argwhere(likelihoods == max_value)[0]
    f1 = np.linspace(-10, 10, 200)[f1]
    f3 = np.linspace(-10, 10, 200)[f3]
    print("max value:", round(max_value, 3), "\nf1:", round(f1, 3), "\nf3:", round(f3, 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
