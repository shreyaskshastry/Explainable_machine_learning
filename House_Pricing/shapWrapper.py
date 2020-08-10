import shap
class ShapWrapper:
    def __init__(
        self,
        predict_fn,
        data,
        sampler=None,
        feature_names=None,
        feature_types=None,
        explain_kwargs=None,
        n_jobs=1,
        **kwargs
    ):

        self.data, _, self.feature_names, self.feature_types = unify_data(
            data, None, feature_names, feature_types
        )
        self.predict_fn = predict_fn(predict_fn, self.data)
        self.n_jobs = n_jobs

        if sampler is not None: 
            warnings.warn("Sampler interface not currently supported.")
        self.sampler = sampler
        self.explain_kwargs = explain_kwargs
        self.kwargs = kwargs

        self.shap = shap.KernelExplainer(self.predict_fn, data, **self.kwargs)
