def mse_loss(y_pred, y_true):
    """
    MSE Loss
    """
    return sum([(tru-prd)**2 for tru, prd in zip(y_true, y_pred)])