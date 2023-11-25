import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        # epsilon value: increases stability for very wrong predictions to prevent values close to log(0)
        self.epsilon = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor

        #Where y_k = 1: -> where label_tensor == 1
        prediction_tensor = prediction_tensor[label_tensor==1]

        # devide the whole calculation into small steps and then compose them
        add = prediction_tensor + self.epsilon
        log = (-1)*np.log(add)
        return np.sum(log)
    
    def backward(self, label_tensor):

        # E_n = - y/(y_hat + epsilon)
        # use the predictions from the forward step
        return (-1)*np.divide(label_tensor, (self.prediction_tensor + self.epsilon))
