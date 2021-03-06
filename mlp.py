import numpy as np
import pickle
import random

class MuliLayerPerceptron:
    
    def sigma(self, z):
        return(1 / (1 + np.exp(-z)))
    
    def d_sigma(self, z):
        s = self.sigma(z)
        return s * (1 - s)
    
    def tanh(self, z):
        return( (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) )
    
    def d_tanh(self, z):
        t = self.tanh(z)
        return 1 - t*t
    
    def ReLU(self, z):
        if z > 0:
            return np.ones(z.shape)
        else:
            return np.zeros(z.shape)
    
    def d_ReLU(self, z):
        if z > 0:
            return z
        else:
            return np.zeros(z.shape)
        
    def softplus(self, z):
        return(np.log( 1 + ))
    
    def d_softplus(self, z):
        s = self.sigma(z)
        return s * (1 - s)
    
    self.act_function_dict = {"sigma": self.sigma,
                              "tanh" : self.tanh,
                              "ReLU" : self.ReLU, 
                              "softplus" : self.softplus}
    
    self.d_act_function_dict = {"sigma": self.d_sigma,
                                "tanh" : self.d_tanh,
                                "ReLU" : self.d_ReLU, 
                                "softplus" : self.d_softplus}

   	def mse(self, y, y_hat):
   		loss = np.sum( (y - y_hat)**2 ) / y.shape[0]
   		return loss

   	def d_mse(self, y, y_hat):
   		d = np.sum( 2 * (y - y_hat)) / y.shape[0]
   		return d

	def softmax(self, y):
	    exps = np.exp(y)
	    return exps / np.sum(exps)

	# TO DO Cross entropy
	def cross_entropy(self, p, y_hat):
	    q = self.softmax(y_hat)
	   	
	    loss = -np.sum(p * np.log(q))
	    return loss

	def d_cross_entropy(self, p, y_hat):
	    q = self.softmax(y_hat)
	   	
	    loss = -np.sum(p * np.log(q))
	    return loss

                              
    def __init__(self, layer_size, act_function, bias, batch_size, epoches, eta, momentum, problem_type, random):
    	"""
        Parameters
        ----------
        layer_size : list
            List with values - number of neurons in each layer, and lenght the number of all layers
        act_function : str
            Activation function used in network
        bias : bool
            If bias should be used in network
        batch_size: int
        	Size of the batch
        epochs: int
        	Number of epochs/iterations of learning
        eta: float
        	Learning rate
        momentum: float
        	Value of momentum in range 0-1 to keep the same direction of learning
        problem_type: str
        	Determines if we are dealing with regression or classification (1 = "regression", anything else/0 ="classification")
        random: bool
        	Parameters which indicates random seed, if tre then we can recreate our results 
        """
        self.layer_size = layer_size
        self.act_function = self.act_function_dict[act_function]
        self.d_act_function = self.d_act_function_dict[act_function]
        self.bias = bias
        self.weights = [np.random.randn( (lambda ls, b: ls+1 if b else ls)(layers_size[i], bias), layers_size[i-1]) for i in range(1, len(layers_size))]
        self.batch_size = batch_size
        self.epoches = epoches
        self.eta = eta
        self.momentum = momentum
        self.loss_function = self.mse if problem_type else cross_entropy
        self.seed = seed
    
    def forward_pass(self, x):
        self.z = []
        last_a = x
        self.a = [last_a]
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, last_a) + b
            self.z.append(z)
            last_a = self.act_function(z)
            self.a.append(last_a)
    
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        self.forward_pass(x)

        delta = (self.a[-1] - y) * self.d_act_function(self.z[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, self.a[-2].T())

        for l in range(2, len(self.layers)):
            delta = np.dot(self.weights[-l+1].T(), delta) * self.d_act_function(self.z[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, self.a[-l-1].T())
        return (nabla_b, nabla_w)
    
    def SGD(self, training_data, test_data):
    	self.train_error = np.zeros((epoches,1))
        self.test_error = np.zeros((epoches,1))
        self.edges_weights = [np.zeros((lambda ls, b: ls+1 if b else ls)(layers_size[i], bias), layers_size[i-1]) for i in range(1, len(layers_size))]
        self.edges_errors = [np.zeros((lambda ls, b: ls+1 if b else ls)(layers_size[i], bias), layers_size[i-1]) for i in range(1, len(layers_size))]
        
    	if self.seed:
    		np.random.seed(111)

        for epoch in range(self.epoches):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+self.batch_size] for k in range(0, len(training_data), self.batch_size)]
            for mini_batch in mini_batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                for x, y in mini_batch:
                    delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                self.weights = [w-(self.eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b-(self.eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
  
            s = 0
            for i,(x, y) in enumerate(test_data):
                self.forward_pass(x)
                s += int(np.argmax(self.a[-1]) == y)
            print("Epoch {} : {} / {}".format(epoch, s, len(test_data)))