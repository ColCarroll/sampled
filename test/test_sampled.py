import numpy as np
import pymc3 as pm
import theano.tensor as tt

from sampled import sampled


def test_sampled_one_model():
    @sampled
    def just_a_normal():
        pm.Normal('x', mu=0, sd=1)

    draws = 50
    with just_a_normal():
        decorated_trace = pm.sample(draws=draws, tune=50, init=None)

    assert decorated_trace.varnames == ['x']
    assert len(decorated_trace.get_values('x')) == draws


def test_reuse_model():
    @sampled
    def two_normals():
        mu = pm.Normal('mu', mu=0, sd=1)
        pm.Normal('x', mu=mu, sd=1)

    with two_normals():
        generated_data = pm.sample(draws=50, tune=50, init=None)

    for varname in ('mu', 'x'):
        assert varname in generated_data.varnames

    with two_normals(mu=1):
        posterior_data = pm.sample(draws=50, tune=50, init=None)

    assert 'x' in posterior_data.varnames
    assert 'mu' not in posterior_data.varnames
    assert posterior_data.get_values('x').mean() > generated_data.get_values('x').mean()


def test_linear_model():
    rows, cols = 1000, 10
    X = np.random.normal(size=(rows, cols))
    w = np.random.normal(size=cols)
    y = X.dot(w) + np.random.normal(scale=0.1, size=rows)

    @sampled
    def linear_model(X, y):
        shape = X.shape
        X = pm.Normal('X', mu=np.mean(X, axis=0), sd=np.std(X, axis=0), shape=shape)
        coefs = pm.Normal('coefs', mu=tt.zeros(shape[1]), sd=tt.ones(shape[1]), shape=shape[1])
        pm.Normal('y', mu=tt.dot(X, coefs), sd=tt.ones(shape[0]), shape=shape[0])

    with linear_model(X=X, y=y):
        sampled_coefs = pm.sample(draws=1000, tune=500)
    mean_coefs = sampled_coefs.get_values('coefs').mean(axis=0)
    np.testing.assert_allclose(mean_coefs, w, atol=0.1)


def test_partial_model():
    rows, cols = 1000, 10
    X = np.random.normal(size=(rows, cols))
    w = np.random.normal(size=cols)
    y = X.dot(w) + np.random.normal(scale=0.1, size=rows)

    @sampled
    def partial_linear_model(X):
        shape = X.shape
        X = pm.Normal('X', mu=np.mean(X, axis=0), sd=np.std(X, axis=0), shape=shape)
        pm.Normal('coefs', mu=tt.zeros(shape[1]), sd=tt.ones(shape[1]), shape=shape[1])

    with partial_linear_model(X=X) as model:
        coefs = model.named_vars['coefs']
        pm.Normal('y', mu=tt.dot(X, coefs), sd=tt.ones(y.shape), observed=y)
        sampled_coefs = pm.sample(draws=1000, tune=500)

    mean_coefs = sampled_coefs.get_values('coefs').mean(axis=0)
    np.testing.assert_allclose(mean_coefs, w, atol=0.1)
