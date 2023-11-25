import numpy as np

class L2_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha * weights

    def norm(self, weights):
        #return self.alpha * (np.linalg.norm(weights, 'fro'))**2  #test:32*1337, here:((5.65)^2 )*1337
        return self.alpha * (np.linalg.norm(weights))**2

class L1_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha*np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.linalg.norm(np.ravel(weights), 1)
