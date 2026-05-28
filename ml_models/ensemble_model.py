import joblib
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge

def build_voting_ensemble(estimators):
    """
    Builds an average voting regressor from a list of (name, estimator) tuples.
    """
    return VotingRegressor(estimators=estimators)

def build_stacking_ensemble(estimators):
    """
    Builds a stacking regressor from a list of (name, estimator) tuples.
    Uses Ridge regression as the final estimator.
    """
    return StackingRegressor(estimators=estimators, final_estimator=Ridge())

def get_weighted_predictions(models, weights, X):
    """
    Generate predictions using custom weights.
    models: list of loaded models
    weights: list of floats summing to 1
    X: features
    """
    if len(models) != len(weights):
        raise ValueError("Number of models must match number of weights")
        
    predictions = 0
    for model, weight in zip(models, weights):
        predictions += weight * model.predict(X)
        
    return predictions
