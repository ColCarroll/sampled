|Build Status| |Coverage Status|

========
sampled
========


*Decorator for reusable models in PyMC3*

Provides syntactic sugar for reusable models with PyMC3.  This lets you separate creating a generative model from using the model.

Here is an example of creating a model:
::

    import numpy as np
    import pymc3 as pm
    from sampled import sampled

    @sampled
    def linear_model(X, y):
        shape = X.shape
        X = pm.Normal('X', mu=np.mean(X, axis=0), sd=np.std(X, axis=0), shape=shape)
        coefs = pm.Normal('coefs', mu=np.zeros(shape[1]), sd=np.ones(shape[1]), shape=shape[1])
        pm.Normal('y', mu=np.dot(X, coefs), sd=np.ones(shape[0]), shape=shape[0])

Now here is how to use the model:
::

    X = np.random.normal(size=(1000, 10))
    w = np.random.normal(size=10)
    y = X.dot(w) + np.random.normal(scale=0.1, size=1000)

    with linear_model(X=X, y=y):
        sampled_coefs = pm.sample(draws=1000, tune=500)

    np.allclose(sampled_coefs.get_values('coefs').mean(axis=0), w, atol=0.1) # True

You can also use this to build graphical networks -- here is a continuous version of the `STUDENT` example from Koller and Friedman's "Probabilistic Graphical Models", chapter 3:
::
    import numpy as np
    import theano.tensor as tt
    import pymc3 as pm
    from sampled import sampled

    @sampled
    def student():
        difficulty = pm.Beta('difficulty', alpha=5, beta=5)
        intelligence = pm.Beta('intelligence', alpha=5, beta=5)
        SAT = pm.Beta('SAT', alpha=20 * intelligence, beta=20 * (1 - intelligence))
        grade_avg = 0.5 + 0.5 * tt.sqrt((1 - difficulty) * intelligence)
        grade = pm.Beta('grade', alpha=20 * grade_avg, beta=20 * (1 - grade_avg))
        recommendation = pm.Binomial('recommendation', n=1, p=0.7 * grade)

Observations may be passed into any node, and we can observe how that changes posterior expectations:

::

    # no prior knowledge
    with student():
        prior = pm.sample(draws=1000, tune=500)

    prior.get_values('recommendation').mean()  # 0.502

    # 99th percentile SAT score --> higher chance of a recommendation
    with student(SAT=0.99):
        good_sats = pm.sample(draws=1000, tune=500)

    good_sats.get_values('recommendation').mean()  # 0.543

    # A good grade in a hard class --> very high chance of recommendation
    with student(difficulty=0.99, grade=0.99):
        hard_class_good_grade = pm.sample(draws=1000, tune=500)

    hard_class_good_grade.get_values('recommendation').mean()  # 0.705


**References**

*  Koller, Daphne, and Nir Friedman. *Probabilistic graphical models: principles and techniques.* MIT press, 2009.

.. |Build Status| image:: https://travis-ci.org/ColCarroll/sampled.svg?branch=master
   :target: https://travis-ci.org/ColCarroll/sampled
.. |Coverage Status| image:: https://coveralls.io/repos/github/ColCarroll/sampled/badge.svg?branch=master
   :target: https://coveralls.io/github/ColCarroll/sampled?branch=master
