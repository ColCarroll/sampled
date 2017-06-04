import pymc3 as pm


class ObserverModel(pm.Model):
    """Stores observed variables until the model is created."""
    def __init__(self, observed):
        self.observed = observed
        super(ObserverModel, self).__init__()

    def Var(self, name, dist, data=None, total_size=None):
        return super(ObserverModel, self).Var(name, dist,
                                              data=self.observed.get(name, data),
                                              total_size=total_size)


def sampled(f):
    """Decorator to delay initializing pymc3 model until data is passed in."""
    def wrapped_f(**observed):
        try:
            with ObserverModel(observed) as model:
                f(**observed)
        except TypeError:
            with ObserverModel(observed) as model:
                f()
        return model
    return wrapped_f
