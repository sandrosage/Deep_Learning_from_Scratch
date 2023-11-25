import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        # self.alpha*np.linalg.norm(weights, ord='fro')
        # return self.alpha * np.sqrt((weights**2).sum())
        return self.alpha * np.linalg.norm(weights)**2
        
        
class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.abs(weights).sum()