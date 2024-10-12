def elastic_net_loss(beta, X, y, alpha=0.001, l1_ratio=0.5):
    """
    Loss function for ElasticNet regression (L1 + L2 regularization)
    beta: coefficients (weights)
    X: feature matrix (with significant features)
    y: target variable
    alpha: regularization strength (controls the penalty)
    l1_ratio: balance between L1 and L2 (1.0 = L1, 0.0 = L2)
    """
    predictions = np.dot(X, beta)  # Predicted values
    residuals = y - predictions     # Error in predictions
    

    l1_penalty = np.sum(np.abs(beta))
    l2_penalty = np.sum(beta**2)
    
    # ElasticNet loss function: squared loss + L1 + L2
    loss = np.sum(residuals**2) / (2 * len(y)) + alpha * (l1_ratio * l1_penalty + (1 - l1_ratio) * l2_penalty)
    return loss
