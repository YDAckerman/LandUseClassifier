import os
import numpy as np
from random import sample
from scipy import io as sio
import theano
import theano.tensor as T

def load_data(train_prop = .8, valid_prop = .2):

    #############
    # LOAD DATA #
    #############
    
    os.chdir('/Users/Yoni')
    
    raw = sio.loadmat("Documents/ZhangLab/LandUseClass/Data/Salinas_corrected.mat")
    gt = sio.loadmat("Documents/ZhangLab/LandUseClass/Data/Salinas_gt.mat")

    raw = raw['salinas_corrected']
    gt = gt['salinas_gt']
    
    vals = []
    targs = []
    
    for i in range(len(raw)):
        for j in range(len(raw[1])):
            targ = gt[i][j]
            if(targ == 0):
                continue
            else:
                vals.append(raw[i][j])
                targs.append(targ - 1)
            
    vals = np.array(vals)
    targs = np.array(targs)

    train_vals = []
    test_vals = []
    valid_vals = []
    train_targs = []
    test_targs = []
    valid_targs = []

    for t in np.unique(targs):
        i = np.where(targs == t)[0]
        i_train = sample(i, 20) #int(train_prop * len(i)))
        i_test = sample(i, 20) #np.setdiff1d(i, i_train)
        i_valid = sample(i, 20) #sample(i_train, int(valid_prop * len(i_train)))
        #i_train = np.setdiff1d(i_train, i_valid)

        train_vals.append(vals[i_train])
        test_vals.append(vals[i_test])
        valid_vals.append(vals[i_valid])

        train_targs.append(targs[i_train])
        test_targs.append(targs[i_test])
        valid_targs.append(targs[i_valid])

    train_vals = np.concatenate(train_vals, axis = 0)
    test_vals = np.concatenate(test_vals, axis = 0)
    valid_vals = np.concatenate(valid_vals, axis = 0)

    train_targs = np.concatenate(train_targs, axis = 0)
    test_targs = np.concatenate(test_targs, axis = 0)
    valid_targs = np.concatenate(valid_targs, axis = 0)
    
    train_set = (train_vals, train_targs)
    valid_set = (valid_vals, valid_targs)
    test_set = (test_vals, test_targs)
    
    def shared_dataset(data_xy, borrow = True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX),
                            borrow = borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype = theano.config.floatX),
                            borrow = borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
