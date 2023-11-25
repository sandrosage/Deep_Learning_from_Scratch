import numpy as np


class CrossEntropyLoss:
    def __init__(self) -> None:
        self.error_tensor = None
        self.loss = None
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        one_ = label_tensor == 1
        self.loss = np.log(prediction_tensor[one_] + np.finfo(float).eps)
        self.loss = -(np.sum(self.loss))
        self.prediction_tensor = prediction_tensor
        return self.loss

    def backward(self, label_tensor):
        E = -label_tensor / (self.prediction_tensor + np.finfo(float).eps)
        return E
