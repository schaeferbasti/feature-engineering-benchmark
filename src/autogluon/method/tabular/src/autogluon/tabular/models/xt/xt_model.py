from src.autogluon.method.core.src.autogluon.core.constants import QUANTILE, REGRESSION

from ..rf.rf_model import RFModel


class XTModel(RFModel):
    """
    Extra Trees model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
    """

    def _get_model_type(self):
        if self.problem_type == REGRESSION:
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor
        elif self.problem_type == QUANTILE:
            from ..rf.rf_quantile import ExtraTreesQuantileRegressor

            return ExtraTreesQuantileRegressor
        else:
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier
