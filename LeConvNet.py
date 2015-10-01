"""class LeNet includes one convolution layer and a single
fully-connected layer. It is adapted from code found at
http://deeplearning.net/tutorial/code/convolutional_mlp.py

notes, the paper guiding this project uses the following
hyper-parameter notation:
n1 : the number of spectral bands
k1 : the kernel size (k1, 1) (they recommend 20 kernels)
n2 : n2 = n1 - k1 + 1, used to calculated the number of nodes in
     the convolution layer (20 x n2, 1)
k2 : kernel size (k2, 1) of the max pooling layer
n3 : n3 = n2 / k2, used to calculate the number of nodes in the
     max pooling layer (20 x n3, 1) (no parameters in this layer)
n4 : number of nodes in the fully connected layer between the max
     pooling layer and the output layer
n5 : number of output nodes (number of classes)
the paper recommends these guidelines:
k1 = n1 / 12 (the paper says to use denom of 9, which they don't explain)
n3 = n2 / k2 (not the way the paper states it, but they do it badly.)
n3 \in [30, 40]
n4 = 100
"These choices might not be the best but are effective for general HSI data"
"""

import os
import sys
import timeit

import numpy as np
from random import sample
from scipy import io as sio

import theano
import theano.tensor as T

from LogitReg import LogisticRegression
from MLPLayer import HiddenLayer
from CNNLayer import LeNetConvPoolLayer
from loadData import load_data


class LeNet(object):

    def __init__(self, rng, input, batch_size, nkerns, n_in,
                 n_hidden, n_out, cost = "softmax"):
        """Initialize the parameters for the multilayer perceptron

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        
        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # hypothetical hyper-parameters (will optimize later?):
        n1 = 204
        k2 = 4

        """
        the given n1 and n3 imply 
        k1 = 17
        n2 = 188
        k2 = 4
        n3 = 47
        n4 = 100
        n5 = 16
        """
        
        ## Create a initial layer of convolution & pooling
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=input,
            image_shape=(batch_size, 1, 204, 1), 
            filter_shape=(nkerns[0], 1, 17, 1), 
            poolsize=(4, 1)
        )

        ## create the next layer as a fully connected mlp
        self.layer1 = HiddenLayer(
            rng=rng,
            input=self.layer0.output.flatten(2),
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # Fully connected Logit layer
        self.layer2 = LogisticRegression(
            input=self.layer1.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.layer0.W).sum()
            + abs(self.layer1.W).sum()
            + abs(self.layer2.W).sum()
            )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.layer0.W ** 2).sum()
            + (self.layer1.W ** 2).sum()
            + (self.layer2.W ** 2).sum()
        )
        
        # NLL of the MLP is computed in the logistic layer (layer2)
        self.negative_log_likelihood = (
            self.layer2.negative_log_likelihood
        )

        # same is true of the errors
        self.errors = self.layer2.errors

        # parameters of the model are the parameters of the layers:
        self.params = self.layer0.params + self.layer1.params + self.layer2.params

        # keep track of the model input
        self.input = input

        # keep track of the predicted value
        self.y_pred = self.layer2.y_pred


def train_LeNet(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=90,
               batch_size=20, nkerns=[20, 50], n_hidden=100):
    
    """
    Stochastic gradient descent optimization for a spectral classifier Convolution
    Neural Network

    :type learning_rate: float
    :param learning_rate: learning rate used (for sgd)

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the dataset file

    :type batch_size: int
    :param batch_size: used to compute the number of mini batches for
    training, validation, and testing.

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type n_hidden: int
    :param n_hidden: sets the number of perceptrons in the hidden layer
    """
    rng = np.random.RandomState(23455)
                  
    datasets = load_data() ## you will need to write this function

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 204, 1))

    classifier = LeNet(
        rng = rng,
        input = layer0_input,
        batch_size = 20,
        nkerns = nkerns,
        n_in = nkerns[0] * 47,
        n_hidden = 100,
        n_out = 16 # number of classes
    )

    cost = classifier.negative_log_likelihood(y) # add L1, L2 to cost?

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        classifier.layer2.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        classifier.layer2.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = classifier.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1 # increment iterator
        for minibatch_index in xrange(n_train_batches): # go through each minibatch

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
                  # notify user after each 100
                  # batches
            
            cost_ij = train_model(minibatch_index)
            # calculate cost with this minibatch

            if (iter + 1) % validation_frequency == 0:
                  # If we've covered enough iterations to give validation
                  # a try, do so:
                  
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    #join lines with \
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            # early exit 
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    train_LeNet(n_epochs = 20)
