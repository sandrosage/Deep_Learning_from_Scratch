import numpy as np
from scipy import signal
from Layers.Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape=1, convolution_shape=None, num_kernels=None) -> None:
        super().__init__()
        self.trainable = True
        self.stride_shape = (stride_shape)
        self.convolution_shape = (convolution_shape)
        self.num_kernels = num_kernels
        


        self.weights = np.random.uniform(0,1,(num_kernels,*convolution_shape))
        
        self.bias = np.random.uniform(0,1,(num_kernels))
        
        #self.weights = np.vstack([self.weights,self.bias])
        self.gradient_weights = None
        self.gradient_bias = None
        self.output_shape = None
        self.output = None
        self._optimizer = None
        self._bias_optimizer = None

    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self._bias_optimizer = value
        
    def initialize(self, weights_initializer,bias_initializer):
        
        self.weights = weights_initializer.initialize((self.num_kernels,*self.convolution_shape),np.prod(self.convolution_shape),np.prod(self.convolution_shape[1:]) * self.num_kernels)
        
        self.bias = bias_initializer.initialize(((self.num_kernels)),np.prod(self.convolution_shape),np.prod(self.convolution_shape[1:]) * self.num_kernels)
       

    def forward(self, input_tensor):

        #output shape of kernel
        #1D case of kernel
        if len(input_tensor.shape) <= 3:
            self.output_shape = (input_tensor.shape[0],self.num_kernels, input_tensor.shape[2])
        #2D case of kernel
        else:
            self.output_shape = (input_tensor.shape[0],self.num_kernels, input_tensor.shape[2] , input_tensor.shape[3])

        self.output_shape = list(self.output_shape)

        if len(self.stride_shape) > 1:
            self.output_shape[2] = np.ceil(self.output_shape[2] / self.stride_shape[0]).astype('int')
            self.output_shape[3] = np.ceil(self.output_shape[3] / self.stride_shape[1]).astype('int')
        else:
            self.output_shape[2] = np.ceil((self.output_shape[2] / self.stride_shape[0])).astype('int')
        
        self.output_shape = tuple(self.output_shape)

        
        self.input_tensor = input_tensor
        output = np.zeros(self.output_shape)
        samples = input_tensor.shape[0]
        channels = input_tensor.shape[1]

        #for every sample
        for b in range(samples):
            #for every kernel
            for k in range(self.num_kernels):
                #for every channel
                for c in range(channels):
                    #print(f'sample #{b},kernel #{k},channel #{c}')
                    if len(self.stride_shape) > 1:
                        output[b,k] += signal.correlate(input_tensor[b,c], self.weights[k,c],'same')[::self.stride_shape[0], ::self.stride_shape[1]] 
                    else:
                        output[b,k] += signal.correlate(input_tensor[b,c], self.weights[k,c],'same')[::self.stride_shape[0]]
                
                output[b,k]+=self.bias[k]
        
        
        self.output = output 

        return output
    

        

    def backward(self,error_tensor):
        samples = self.input_tensor.shape[0]
        Ex = None

        #1D case
        if len(self.input_tensor.shape) <= 3:
            _, channels, x=self.input_tensor.shape
            Ex=np.zeros((samples, self.convolution_shape[0],x))
            self.gradient_weights=np.zeros((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]))
            
            for i in range(samples):
                img=error_tensor[i,:,:]
                strideMatrix = np.zeros ((self.stride_shape[0]))
                strideMatrix[0]=1
                img=np.kron(img,strideMatrix)
                img=img [:,0:x]
                pad1 = (self.convolution_shape [1]-1) //2, self.convolution_shape [1]//2
                
                for m in range(self.num_kernels):
                    gradient_tmp=np.zeros((self.convolution_shape[0], self.convolution_shape[1])) 
                    for l in range(channels):
                        inputPad = np.pad (self.input_tensor [i, l, :], pad1)
                        gradient_tmp[l,:]+= signal.correlate(inputPad, img[m,:], mode='valid') 
                    self.gradient_weights [m, :,:]= gradient_tmp
                
                correlation=np.zeros((channels,x)) 
                
                for j in range(channels):
                    corr=np.zeros((x))
                    for k in range(self.num_kernels):
                        corr+=signal.convolve(img [k,:],self.weights[k,j,:],mode= 'same') 
                    correlation[j,:]=corr
            Ex [i,:,:]=correlation
        
            self.gradient_bias = np.sum(error_tensor, axis= (0,2))
        #2D Case
        else:
            _,channels,y,x=self.input_tensor.shape
            
            Ex=np.zeros((samples,self.convolution_shape[0],y,x))

            self.gradient_weights=np.zeros((self.num_kernels,)+self.convolution_shape)
            
            for i in range(samples):
                img=error_tensor[i,:,:,:]
                strideMatrix = np.zeros((self.stride_shape[0],self.stride_shape[1]))
                strideMatrix [0,0]=1
                img=np.kron(img,strideMatrix) 
                img=img [:,0:y, 0:x]
                pad1 = (self.convolution_shape[1]-1)//2, self.convolution_shape[1]//2
                pad2 = (self.convolution_shape[2]-1)//2, self.convolution_shape[2]//2
                
                for m in range(self.num_kernels) :
                    gradient_tmp=np.zeros((self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2])) 
                    for l in range(channels):
                        inputPad = np.pad(self.input_tensor [i,l,:,:], (pad1, pad2))
                        gradient_tmp[l,:,:]+=signal.correlate2d(inputPad, img[m,:,:], mode= 'valid') 
                    self.gradient_weights[m,:, :,:]+=gradient_tmp
                
                correlation=np.zeros((channels,y,x))
                for j in range(channels):
                    corr=np.zeros((y,x))
                    for k in range(self.num_kernels):
                        corr+=signal.convolve(img[k,:,:],self.weights[k,j,:,:], mode='same') 
                    correlation[j,:,:]=corr
                Ex[i,:,:,:]=correlation
            self.gradient_bias = np.sum(error_tensor, axis= (0,2,3)) 

        
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)


        return Ex

