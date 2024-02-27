def mse_loss(y_pred, y_true):
    """
    MSE Loss
    """
    diff = [(tru-prd) for tru, prd in zip(y_true, y_pred)]
    diff_sq = [d*d for d in diff]
    v1 = diff_sq[0]
    for val in diff_sq[1:]:
        v1 += val
    return v1