from abc import ABC, abstractmethod

class BaseLayer(ABC):
    def __init__(self) -> None:
        self.trainable = False
        #Optional weights parameter?
        self.weights = None
    
    @abstractmethod
    def forward(self,input_tensor):
        raise NotImplementedError("Forward Method not Implented")
    
    @abstractmethod
    def backward(self,error_tensor):
        raise NotImplementedError("Backward Method not Implented")