class Sgd:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    # gradient descent calculation
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate*(gradient_tensor)