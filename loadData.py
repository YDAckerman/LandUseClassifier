import os
import numpy as np
# from random import sample, shuffle
from scipy import io as sio
import theano
import theano.tensor as T

def load_data(rng):

    #############
    # LOAD DATA #
    #############
    
    os.chdir(os.path.expanduser("~"))
    
    raw = sio.loadmat("Documents/ZhangLab/Python/LandUseClassifier/Data/Salinas_corrected.mat")
    gt = sio.loadmat("Documents/ZhangLab/Python/LandUseClassifier/Data/Salinas_gt.mat")

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

    train_vals, test_vals, valid_vals = [], [], []
    train_targs, test_targs, valid_targs = [], [], []


    for t in np.unique(targs):
        i = np.where(targs == t)[0]
        i_train = rng.choice(i, 400) #int(.2 * len(i)))
        i = np.setdiff1d(i, i_train)
        i_valid = rng.choice(i, 200) #int(.2 * len(i)))
        i_test = np.setdiff1d(i, i_valid)
        train_vals = train_vals + [vals[j] for j in i_train]
        test_vals = test_vals + [vals[j] for j in i_test]
        valid_vals = valid_vals + [vals[j] for j in i_valid]
        train_targs = train_targs + [targs[j] for j in i_train]
        test_targs = test_targs + [targs[j] for j in i_test]
        valid_targs = valid_targs + [targs[j] for j in i_valid]

    # for some reason I'm concerned that having it ordered as it is will cause
    # problems... let's shuffle it all up

    i_train = range(len(train_targs))
    i_valid = range(len(valid_targs))
    i_test = range(len(test_targs))

    rng.shuffle(i_train)
    rng.shuffle(i_valid)
    rng.shuffle(i_test)
    
    train_vals = [train_vals[i] for i in i_train]
    train_targs = [train_targs[i] for i in i_train]

    valid_vals = [valid_vals[i] for i in i_valid]
    valid_targs = [valid_targs[i] for i in i_valid]

    test_vals = [test_vals[i] for i in i_test]
    test_targs = [test_targs[i] for i in i_test]
    
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
