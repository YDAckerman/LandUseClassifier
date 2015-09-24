import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from CNNLayer import LeNetConvPoolLayer
from MLPLayer import HiddenLayer
from LogitReg import LogisticRegression
from LeConvNet import LeNet
from loadData import load_data


############################################################################
rng = np.random.RandomState(23455)

input = T.tensor4(name='input')
y = T.ivector('y')

theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'

cnnlayer = LeNetConvPoolLayer(
    rng = rng,
    input = input,
    filter_shape = (20, 1, 17, 1),
    image_shape = (1, 1, 204, 1)
)

hiddenlayer = HiddenLayer(
    rng,
    input = cnnlayer.output.flatten(2),
    n_in = 940,
    n_out = 100,
    activation = T.tanh
)

logisticlayer = LogisticRegression(
    input = hiddenlayer.output,
    n_in = 100,
    n_out = 16
)

cost = logisticlayer.negative_log_likelihood(y)
errors = logisticlayer.errors(y)                                             
g = theano.function(inputs = [input, y], outputs = errors)
predict = theano.function(inputs = [input], outputs = logisticlayer.y_pred)


# dataset = load_data()
# train_set_x, train_set_y = dataset[0]
# layer_input = train_set_x[0].reshape(1,1,204,1)
x = np.random.random_integers(190, size=(204.,1.)).reshape(1,1,204,1)
target = np.asarray([1], 'int32')
filtered_layer = g(x,target)
pred = predict(x)


# Works! I didn't: have the right number of input nodes to the
# hidden layer; and I didn't have target as a single integers. ok!

################################################################################

batch_size = 20
learning_rate = 0.01
nkerns = (20,50)

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
    updates = updates,
    givens = {
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

## for when we want to output predictions
predict = theano.function(
    inputs = [index],
    outputs = classifier.y_pred,
    givens = {
        x: train_set_x[index * batch_size: (index + 1) * batch_size]
    }
)

# proof that the problem was in the values of the y_train vector (cnet considered
# the classes 0-15, but they were labeled 1-16...)
tmp = theano.function([index], outputs = train_set_y[index * 20: (index+1) * 20])

cost_ij = train_model(15)
preds = predict(1)
