def elastic_net_loss(beta, X, y, alpha=0.001, l1_ratio=0.5):
    ...
loss = np.sum(residuals**2) / (2 * len(y)) + alpha * (l1_ratio * l1_penalty + (1 - l1_ratio) * l2_penalty)
    return loss
result = minimize(elastic_net_loss, initial_beta, args=(X, y, alpha, l1_ratio), method='BFGS')
