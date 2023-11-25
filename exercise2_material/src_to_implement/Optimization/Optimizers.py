import numpy as np
class Sgd:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    # gradient descent calculation
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate*(gradient_tensor)
    
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v_k = (self.momentum_rate*self.v_k) -(self.learning_rate*gradient_tensor)
        return weight_tensor + self.v_k


class Adam:
    def __init__(self, learning_rate, mu, rho):
        # ß1 -> mu and ß2 -> rho
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k = 0
        self.r_k = 0

        # exponent for mu and rho in the bias correction
        self.k =  1
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        g_k = gradient_tensor
        self.v_k = (self.mu*self.v_k) + (1 - self.mu)*g_k
        self.r_k = (self.rho*self.r_k) + (1- self.rho)*g_k*g_k

        # bias correction
        v_hat = self.v_k/(1 - self.mu**self.k)
        r_hat = self.r_k/(1 - self.rho**self.k)

        # increment exponent
        self.k += 1

        return weight_tensor - self.learning_rate*(v_hat/(np.sqrt(r_hat)+ np.finfo(float).eps))