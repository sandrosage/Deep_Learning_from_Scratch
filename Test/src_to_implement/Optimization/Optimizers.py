import numpy as np


class Sgd:
    def __init__(self, learning_rate:float) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor,gradient_tensor):
        #following formula from slides w(k+1) = w(k) - lr * gradient( L( w(k) ) )   
        gradient_tensor = np.atleast_1d(gradient_tensor)
        #randomly select one gradient from the list of batch gradients
        #grad = np.random.choice(gradient_tensor.shape[0], size=1, replace=False)
        #calculate new weights from random gradient
        new_weights = np.atleast_1d(weight_tensor) - self.learning_rate * gradient_tensor#[grad]        

        return np.atleast_1d(new_weights)

class SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum_rate: float) -> None:
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_vk = 0

    def calculate_update(self,weight_tensor,gradient_tensor):
        gradient_tensor = np.atleast_1d(gradient_tensor)
        weight_tensor = np.atleast_1d(weight_tensor)
        
        v_k = self.momentum_rate * self.prev_vk - self.learning_rate * gradient_tensor
        new_weights = weight_tensor + v_k

        self.prev_vk = v_k
        return np.atleast_1d(new_weights)

class Adam:
    def __init__(self, learning_rate, mu, rho) -> None:
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        
        self.prev_vk = 0
        self.prev_rk = 0
        
        self.count = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.count+=1
        gradient_tensor = np.atleast_1d(gradient_tensor)
        weight_tensor = np.atleast_1d(weight_tensor)
        g_k = gradient_tensor
        
        v_k = self.mu * self.prev_vk + (1-self.mu) * g_k
        r_k = self.rho * self.prev_rk + (1-self.rho) * (g_k * g_k)        

        self.prev_rk = r_k
        self.prev_vk = v_k

        v_k_hat = v_k / (1-self.mu**self.count)
        r_k_hat = r_k / (1-self.rho**self.count)

        new_weights = weight_tensor - self.learning_rate * ( (v_k_hat) / ( np.sqrt(r_k_hat) + np.finfo(float).eps ) )
        
        
        return new_weights


