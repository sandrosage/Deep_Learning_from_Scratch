import numpy as np
from Layers.Javier import Javier_Conv
from Layers.Conv import Conv
from Layers.Initializers import Constant

cov = Conv((1,1),(3,3),10)
test = np.random.uniform(0,1,(4,6,8))
print(cov.weights.shape)
print(cov.bias.shape)
cov.forward(test)