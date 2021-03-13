import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict
import gzip
import os
import pickle

#Layers implementations

class LinearLayer():
    def __init__(self, n_inputs, n_units, rng, bias, name):
        """
        Linear (dense, fully-connected) layer.
        Parameters
        ----------
        n_inputs : int
        n_units : int
            Number of neurons
        rng: float
            random number generator used for initialization
        bias : bool
            Apperance of bias
        name: str
        """
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.rng = rng
        self.name = name
        self.initialize(bias)

    def has_params(self):
        return True

    def forward(self, X):
        """
        Forward message.
        Parameters
        ----------
        X : array
            layer inputs, shape (n_samples, n_inputs)
        Return
        ----------
            : array 
            layer output, shape (n_samples, n_units)
        """
        X2 = X.dot(self.W) + self.b 
        return X2  

    def delta(self, Y, delta_next):
        """
        Computes delta (dl/d(layer inputs)), based on delta from the following layer. The computations involve backward
        message.
        Parameters
        ----------
        Y : array
            output of this layer (i.e., input of the next), shape (n_samples, n_units)
        delta_next : array
            delta vector backpropagated from the following layer, shape (n_samples, n_units)
        Return
        ----------
            : array
            delta vector from this layer, shape (n_samples, n_inputs)
        """
        d = delta_next.dot(self.W.T)
        return d

    def grad(self, X, delta_next):
        """
        Gradient averaged over all samples. The computations involve parameter message.
        Parameters
        ----------
        X : array
            layer input, shape (n_samples, n_inputs)
        delta_next : array
            delta vector backpropagated from the following layer, shape (n_samples, n_units)
        Return
        ----------
            : list 
            a list of two arrays [dW, db] corresponding to gradients of loss w.r.t. weights and biases, the shapes
            of dW and db are the same as the shapes of the actual parameters (self.W, self.b)
        """
        dW = X.T.dot(delta_next)/X.shape[0]
        db = np.average(delta_next, axis=0)

        return [dW, db]

    def initialize(self, bias):
        """
        Perform He's initialization (https://arxiv.org/pdf/1502.01852.pdf). This method is tuned for ReLU activation
        function. Biases are initialized to 1 increasing probability that ReLU is not initially turned off.
        Parameters
        ----------
        bias : bool
            Apperance of bias
        """
        scale = np.sqrt(2.0 / self.n_inputs)
        self.W = self.rng.normal(loc=0.0, scale=scale, size=(self.n_inputs, self.n_units))
        self.b = np.ones(self.n_units) if bias else np.zeros(self.n_units)

    def update_params(self, dtheta):
        """
        Updates weighs and biases.
        Parameters
        ----------
        dtheta : list
            contains a two element list of weight and bias updates the shapes of which corresponds to self.W
            and self.b
        """
        assert len(dtheta) == 2, len(dtheta)
        dW, db = dtheta
        assert dW.shape == self.W.shape, dW.shape
        assert db.shape == self.b.shape, db.shape
        self.W += dW
        self.b += db


class ReLULayer():
    def __init__(self, name):
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        """
        Forward message.
        Parameters
        ----------
        X : array
            layer inputs, shape (n_samples, n_inputs)
        Return
        ----------
            : array 
            layer output, shape (n_samples, n_units)
        """
        X2 = X
        X2[X2 < 0] = 0
        return X2

    def delta(self, Y, delta_next):
        """
        Computes delta (dl/d(layer inputs)), based on delta from the following layer. The computations involve backward
        message.
        Parameters
        ----------
        Y : array
            output of this layer (i.e., input of the next), shape (n_samples, n_units)
        delta_next : array
            delta vector backpropagated from the following layer, shape (n_samples, n_units)
        Return
        ----------
            : array
            delta vector from this layer, shape (n_samples, n_inputs)
        """
        d = np.zeros(Y.shape)
        d[Y>0] = 1
        return d*delta_next


class TanhLayer():
    def __init__(self, name):
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        """
        Forward message.
        Parameters
        ----------
        X : array
            layer inputs, shape (n_samples, n_inputs)
        Return
        ----------
            : array 
            layer output, shape (n_samples, n_units)
        """
        X2 = 2 / (1 + np.exp(-2 * X)) - 1
        return X2

    def delta(self, Y, delta_next):
        """
        Computes delta (dl/d(layer inputs)), based on delta from the following layer. The computations involve backward
        message.
        Parameters
        ----------
        Y : array
            output of this layer (i.e., input of the next), shape (n_samples, n_units)
        delta_next : array
            delta vector backpropagated from the following layer, shape (n_samples, n_units)
        Return
        ----------
            : array
            delta vector from this layer, shape (n_samples, n_inputs)
        """
        d = 1 - (2 / (1 + np.exp(-2 * Y)) - 1) ** 2
        return d*delta_next


class SigmaLayer():
    def __init__(self, name):
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        """
        Forward message.
        Parameters
        ----------
        X : array
            layer inputs, shape (n_samples, n_inputs)
        Return
        ----------
            : array 
            layer output, shape (n_samples, n_units)
        """
        X2 = 1 / ( 1 + np.exp(-X) )
        return X2

    def delta(self, Y, delta_next):
        """
        Computes delta (dl/d(layer inputs)), based on delta from the following layer. The computations involve backward
        message.
        Parameters
        ----------
        Y : array
            output of this layer (i.e., input of the next), shape (n_samples, n_units)
        delta_next : array
            delta vector backpropagated from the following layer, shape (n_samples, n_units)
        Return
        ----------
            : array
            delta vector from this layer, shape (n_samples, n_inputs)
        """
        d = np.exp(-Y) / ( 1 + np.exp(-Y) )**2
        return d*delta_next


class SoftmaxLayer():
    def __init__(self, name):
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        """
        Forward message.
        Parameters
        ----------
        X : array
            layer inputs, shape (n_samples, n_inputs)
        Return
        ----------
            : array 
            layer output, shape (n_samples, n_units)
        """
        X1 = X - np.max(X, axis = 1, keepdims = True)
        X2 = np.exp(X1)
        X2 /= np.sum(X2, axis = 1, keepdims = True)
        return X2

    def delta(self, Y, delta_next):  
        ones = np.ones((Y.shape[0],Y.shape[1],Y.shape[1]))
        eyes = np.repeat([np.eye(Y.shape[1])],Y.shape[0],axis=0)
        y_ones = Y.reshape(Y.shape[0],Y.shape[1],1)*ones
        d = eyes - y_ones
        d = d*Y.reshape(Y.shape[0],1,Y.shape[1])*delta_next.reshape(Y.shape[0],1,Y.shape[1])
        d = np.sum(d, axis=-1)
        return d


class LossCrossEntropy():
    def __init__(self, name):
        self.name = name

    def forward(self, X, T):
        """
        Forward message.
        Parameters
        ----------
        X : array
            loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
            the number of classes
        T : array
            one-hot encoded targets, shape (n_samples, n_inputs)
        Return
        ----------
            : array 
            layer output, shape (n_samples, 1)
        """
        X = np.log(X)
        X2 = X[T>0]
        return -X2.reshape(X2.shape[0],1)


    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        Parameters
        ----------
        X : array
            loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
            the number of classes
        T : array
            one-hot encoded targets, shape (n_samples, n_inputs)
        Return
        ----------
            : array
            delta vector from this layer, shape (n_samples, n_inputs)
        """
        d = T/X
        return -d


class LossMeanSquareError():
    def __init__(self, name):
        self.name = name

    def forward(self, X, Y):
        """
        Forward message.
        """
        X2 = np.sum( (X - Y) ** 2) / X.shape[0]
        return X2


    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        """
        d = 2 * np.sum( (X - T), axis = 1, keepdims = True) / X.shape[0]
        return d


class LossMeanAbsoluteError():
    def __init__(self, name):
        self.name = name

    def forward(self, X, Y):
        """
        Forward message.
        """
        X2 = np.sum(np.abs(X - Y)) / X.shape[0]
        return X2


    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        """
        d = np.zeros(T.shape)
        d[X > T] = 1 / X.shape[0]
        d[X <= T] = -1 / X.shape[0]
        return d


class LossCrossEntropyForSoftmaxLogits():
    def __init__(self, name):
        self.name = name

    def forward(self, X, T):
        X1 = X - np.max(X, axis = 1, keepdims = True)
        X2 = np.sum(np.exp(X1), axis = 1)
        X3 = -X1[T>0] + np.log(X2)
        X3 = X3.reshape(X3.shape[0],1)
        return X3

    def delta(self, X, T):
        X2 = X - T
        return X2


# Multi Layer Perceptron

class MLP():
    def __init__(self, rng, n_inputs, layers=None, layers_default=None, bias=True, batch_size=100, n_epochs=500, eta=0.5, momentum=0.5, classification=True, loss=None, output_layer=None):
        """
        Parameters
        ----------
        rng: float
            Random state
        n_inputs : int
        layers : list
            List of layers (if None then initialize all layers the same)
        layers_default: list
            Parametrs of layers [n_layers, n_neurons, n_outputs, layer_types = activation_function]
        bias : bool
            If bias should be used in network
        batch_size: int
            Size of the batch
        n_epochs: int
            Number of epochs/iterations of learning
        eta: float
            Learning rate
        momentum: float
            Value of momentum in range 0-1 to keep the same direction of learning
        classification: bool
            Determines if we are dealing with regression or classification
        loss:
            Which loss to use. Overrides classification flag effect.
        """
        self.n_inputs = n_inputs
        if layers is not None:
            self.layers = layers
        else:
            n_layers, n_neurons, n_outputs, layer_types = layers_default
            tmp_layers = [LinearLayer(n_inputs=n_inputs, n_units=n_neurons, rng=rng, bias=bias, name='Linear_1')]
            for nl in range(n_layers-1):
                tmp_layers.append(layer_types(name='ActivationFunction_'+str(nl+1)))
                nn = n_neurons if  nl+2 != n_layers else n_outputs
                tmp_layers.append(LinearLayer(n_inputs=n_neurons, n_units=nn, rng=rng, bias=bias, name='Linear_'+str(nl+2)))
            self.layers = tmp_layers
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eta = eta
        self.momentum = momentum
        self.classification = classification
        if loss:
            self.loss = loss
            self.output_layers = [output_layer] if output_layer else []
        else:
            if classification:
                self.loss = LossCrossEntropyForSoftmaxLogits(name='CE')
                self.output_layers = [SoftmaxLayer(name='Softmax_OUT')]
            else:
                self.loss = LossMeanSquareError(name='MSE')
                self.output_layers = []

        self.first_param_layer = self.layers[-1]
        for l in self.layers:
            if l.has_params():
                self.first_param_layer = l
                break

    def propagate(self, X, output_layers=True, last_layer=None):
        """
        Feedforwad network propagation
        Parameters
        ----------
        X : array
            input data, shape (n_samples, n_inputs)
        output_layers : bool 
            controls whether the self.output_layers are appended to the self.layers in evaluatin
        last_layer : layer 
            if not None, the propagation will stop at layer with this name
        Return
        ----------
         : array
            propagated inputs, shape (n_samples, n_units_of_the_last_layer)
        """
        layers = self.layers + (self.output_layers if output_layers else [])
        if last_layer is not None:
            assert isinstance(last_layer, basestring)
            layer_names = [layer.name for layer in layers]
            layers = layers[0: layer_names.index(last_layer) + 1]
        for layer in layers:
            X = layer.forward(X)
        return X

    def evaluate(self, X, T):
        """
        Computes loss.
        Parameters
        ----------
        X : array
            input data, shape (n_samples, n_inputs)
        T : array 
            target labels, shape (n_samples, n_outputs)
        Return
        ----------
         : array
            value of loss function
        """
        return self.loss.forward(self.propagate(X, output_layers=False), T)

    def gradient(self, X, T, prev_grad):
        """
        Computes gradient of loss w.r.t. all network parameters.
        Parameters
        ----------
        X : array
            input data, shape (n_samples, n_inputs)
        T : array 
            target labels, shape (n_samples, n_outputs)
        prev_grad: array
            previous gradient [for momentum]
        Return
        ----------
         : dict
            a dict of records in which key is the layer.name and value the output of grad function
        """
        # TO ADD MOMENTUM
        out={}
        X = [X]
        lay = self.layers + self.output_layers
        for l in lay:
            X.append(l.forward(X[-1]))

        d = self.loss.delta(X[-1], T )

        Y = X[-1]
        for l, x in zip( reversed(lay), reversed(X[:-1])):
            if l.has_params():
                dW, db = l.grad(x, d)
                dW = dW * (1 - self.momentum) + prev_grad[l.name][0] * self.momentum
                db = db * (1 - self.momentum) + prev_grad[l.name][1] * self.momentum
                out[l.name] = [dW, db]
            d = l.delta(Y, d)
            Y = x
        return out

    def accuracy(self, Y, T):
        p = np.argmax(Y, axis=1)
        t = np.argmax(T, axis=1)
        return np.mean(p == t)


    def train(self, X_train, T_train, X_test=None, T_test=None, verbose=False):
        """
        Trains a network using vanilla gradient descent.
        Parameters
        ----------
        X_train : array
        T_train : array
        X_test : array
        T_test : array
        verbose : bool 
            prints evaluation for each epoch if True
        Return
        ----------
         : list
            returns info anout run
        """
        n_samples = X_train.shape[0]
        assert T_train.shape[0] == n_samples
        assert self.batch_size <= n_samples
        run_info = defaultdict(list)

        def process_info(epoch):
            loss_test, acc_test = np.nan, np.nan
            Y = self.propagate(X_train)
            loss_train = self.loss.forward(Y, T_train)
            if self.classification:
                acc_train = self.accuracy(Y, T_train)
                run_info['acc_train'].append(acc_train)
            run_info['loss_train'].append(loss_train)
            if X_test is not None:
                Y = self.propagate(X_test)
                loss_test = self.loss.forward(Y, T_test)
                if self.classification:
                    acc_test = self.accuracy(Y, T_test)
                run_info['loss_test'].append(loss_test)
                run_info['acc_test'].append(acc_test)
            
            if verbose:
                if self.classification:
                    print('epoch: {}, loss: {}/{} accuracy: {}/{}'.format(epoch, np.mean(loss_train), np.nanmean(loss_test),
                                                                        np.nanmean(acc_train), np.nanmean(acc_test)))
                else:
                    print('epoch: {}, loss: {}/{}'.format(epoch, np.mean(loss_train), np.nanmean(loss_test)))

        process_info('initial')
        
        initial_w = {}
        grads = {}
        epoch_grad_size = defaultdict(list)
        for layer in self.layers:
            if layer.has_params():
                initial_w[layer.name] = np.mean(np.abs(layer.W))
                grads[layer.name] = [np.zeros(layer.W.shape), np.zeros(layer.b.shape)]

        for epoch in tqdm(range(1, self.n_epochs + 1)):
            for layer in self.layers:
                    if layer.has_params():
                        epoch_grad_size[layer.name] = [np.zeros(layer.W.shape), np.zeros(layer.b.shape)]

            offset = 0
            while offset < n_samples:
                last = min(offset + self.batch_size, n_samples)
                grads = self.gradient(np.asarray(X_train[offset:last]), np.asarray(T_train[offset:last]), grads)
                for layer in self.layers:
                    if layer.has_params():
                        gs = grads[layer.name]
                        epoch_grad_size[layer.name] += [[np.abs(gs[0]), np.abs(gs[1])]]
                        dtheta = [-self.eta * g for g in gs]
                        layer.update_params(dtheta)

                offset += self.batch_size

            for layer in self.layers:
                if layer.has_params():
                    mean_weight = np.mean(np.abs(layer.W))
                    run_info[f'{layer.name}_mean_weight'].append(mean_weight/initial_w[layer.name])
                    mean_grad_W = np.mean([i[0] for i in epoch_grad_size[layer.name]])
                    run_info[f'{layer.name}_mean_update'].append(np.mean(mean_grad_W))

            process_info(epoch)
        return run_info


# ---------------------------------------
# -------------- EXPERIMENTS ------------
# ---------------------------------------

def plot_convergence(run_info):
    plt.plot(run_info['acc_train'], label='train')
    plt.plot(run_info['acc_test'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.legend()


def plot_test_accuracy_comparison(run_info_dict):
    keys = sorted(run_info_dict.keys())
    for key in keys:
        plt.plot(run_info_dict[key]['acc_test'], label=key)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.legend()

def plot_weight(run_info, layers):
    for layer in layers:
        if layer.has_params():
            plt.plot(run_info[layer.name], label=layer.name)
        plt.xlabel('epoch')
        plt.ylabel('normalized mean absolute weight')
        plt.legend()
    plt.ylim(bottom=0.98, top = 1.04)
    plt.tight_layout()


def experiment_MNIST():
    X_train, T_train, X_test, T_test = load_MNIST()
    np.seterr(all='raise', under='warn', over='warn')
    rng = np.random.RandomState(1234)
    net = MLP(n_inputs=28 * 28,
              layers=[
                  LinearLayer(n_inputs=28 * 28, n_units=64, rng=rng, name='Linear_1'),
                  ReLULayer(name='ReLU_1'),
                  LinearLayer(n_inputs=64, n_units=64, rng=rng, name='Linear_2'),
                  ReLULayer(name='ReLU_2'),
                  LinearLayer(n_inputs=64, n_units=64, rng=rng, name='Linear_3'),
                  ReLULayer(name='ReLU_3'),
                  LinearLayer(n_inputs=64, n_units=64, rng=rng, name='Linear_4'),
                  ReLULayer(name='ReLU_4'),
                  LinearLayer(n_inputs=64, n_units=64, rng=rng, name='Linear_5'),
                  ReLULayer(name='ReLU_5'),
                  LinearLayer(n_inputs=64, n_units=10, rng=rng, name='Linear_OUT'),
              ],
              loss=LossCrossEntropyForSoftmaxLogits(name='CE'),
              output_layers=[SoftmaxLayer(name='Softmax_OUT')]
              )

    run_info = train(net, X_train, T_train, batch_size=3000, eta=1e-1, X_test=X_test, T_test=T_test, n_epochs=100,
                     verbose=True)
    plot_convergence(run_info)
    plt.show()
    plot_weight(run_info, net.layers)
    plt.show()

    with open('MNIST_run_info.p', 'wb') as f:
        pickle.dump(run_info, f)


if __name__ == '__main__':
    #experiment_XOR()

    #experiment_spirals()

    experiment_MNIST()
