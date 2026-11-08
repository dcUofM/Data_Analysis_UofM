"""
    - multi-layer perceptron with one hidden layer
      but arbitrary hidden units
    1) forward propagation to compute output
    2) Calculate errors based on quadratic cost function
        and sigmoidal activation (inverse logit).
        Cost function minimization by maximizing -log(L)
    3)  Learning by back-propagation:
        Back-propagate errors by computing partial derivatives dJ/dw
        and adjust weights

    - source code for MLP from:
      github.com/rasbt/python-machine-;earning-book-3rd-edition

    - data from: https://archive.ics.uci.edu/ml/,
               machine-learning-databases/iris/iris.data
"""

import numpy as np
import sys

class NeuralNetMLP( object):
    """
    Feedforward neural network / Multi-layer perceptron

    Parameters
    ---------------
    d_hidden : int (default: 30) - number of hidden units
    L2       : float (default: 0.)
                regularization parmeter- lambda in lecture notes
    n_iter   : int (default: 100)
               number iterations over training data
    eta      : float (default: 1e-3)
                learning rate
    shuffle  : bool (default : True)
               randomize after each epoch
    minibatch_size : int (default : 1)
               Number of training examples per minibatch
    seed     : int (default : None)
               random seed for shuffling

    Attributes
    ----------------
    dEval     : dict - contains cost fct. values, training accuracy, and validation
                accuracy for each epoch

    """

    def __init__(self, d_hidden=30, L2= 0, n_iter=100, eta=1e-3, shuffle = True, minibatch_size=1, seed=None):
        self.random   = np.random.RandomState( seed)
        self.d_hidden = d_hidden
        self.L2       = L2
        self.n_iter   = n_iter
        self.eta      = eta
        self.shuffle  = shuffle
        self.minibatch_size = minibatch_size # sample window for forward propagation
        self.dEval    = {'cost': [], 'acc_train': [], 'acc_valid': []}

    def onehot(self, a_y):
        """ Encode class labels in y into one-hot representation
        Parameters
        -----------
        y : array, shape = [n_examples]
            target classes

        :return
        ----------
        onehot : array, shape = (n_examples, n_labels)

        """
        a_lab = np.unique(a_y)
        n_classes = len(a_lab)
        m_y_int = np.zeros((n_classes, a_y.shape[0]))
        for i in range(n_classes):
            m_y_int[i, a_lab[i] == a_y] = 1
        # print( m_y_int)
        # print( n_classes, m_y_int.shape[0])
        # print( gregre)
        return m_y_int.T
        # m_OH = np.zeros(( n_classes, a_y.shape[0]))
        # for idx, val in enumerate( a_y.astype( int)):
        #     m_OH[val, idx] = 1.
        # print( n_classes, m_OH.shape[0])
        # print(gergre)
        # return m_OH.T

    def _sigmoid(self, z):
        """ logistic function sigmoid = logit-1 between -250 ~ 0 and 250 ~ 1"""
        return 1./( 1. + np.exp( -np.clip(z, -250, 250)))

    def _sigmoid_gradient(self, z):
        """Compute gradient of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _forward(self, mX):
        """ forward propagate mX through Neural Net """
        # step1: net input of hidden layer
        # Dim: [n_instance, m_feature] dot [m_feature, d_hidden] = [n_instance, d_hidden]
        z_h = np.dot( mX, self.w_h) + self.b_h #?? bias value for each layer
        # step 2: activation of hidden layer
        a_h = self._sigmoid( z_h)

        #step 3: net input for output layer
        # [n_instance, d_hidden] dot [n_hidden, n_classlables] = [n_examples, n_classlabels]
        z_out = np.dot( a_h, self.w_out) + self.b_out

        # step 4: activation of output layer
        a_out = self._sigmoid(z_out)
        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, a_out):
        """
        Parameters
        ---------------
        :param y:   - array shape = [n_instance, n_target]
                    - binary encoded class labels - one hot

        :param a_out: - array shape = [n_instance, n_output_units]

        :return:
        cost:   float (regularized) cost

        """
        #Eq. 3 page 413 in Rascka - penalty term as sum of squared weights
        L2_term = self.L2 * (np.sum( self.w_h**2) + np.sum(self.w_out**2))
        # Eq. 4 on page 414: cost fct. from logistic equation (max. log-likeli of sigmoid)
        term1  = y_enc * np.log( a_out + 1e-5)#avoid log(0)
        term2  = (1. - y_enc) * np.log( 1. - a_out + 1e-5)
        return -np.sum( term1 + term2) + L2_term

        # L2_term = (self.L2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
        # term1 = -y_enc * (np.log(a_out))
        # term2 = (1. - y_enc) * np.log(1. - a_out)
        # cost = np.sum(term1 - term2) + L2_term
        # return cost

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        :parameter
        --------------------

        :param X_train:     training data instances
        :param y_train:     class labels in training set (must be integers!)
        :param X_valid:     validation during training
        :param y_valid:     class labels in validation data (must be integers!)

        :return:
        self    - MLP object
        """
        # n unique class labels
        n_output   = np.unique( y_train).shape[0]
        n_features = X_train.shape[1]

        #------------------------------------------------------
        #           step A: weight initialization
        # ------------------------------------------------------
        # weights to map input to hidden
        self.b_h = np.zeros( self.d_hidden)
        # first weight matrix -  determines size of a_h
        self.w_h = self.random.normal(loc=0.0,scale=0.1,size=(n_features, self.d_hidden))#np.zeros( )

        # weights hidden -> output
        self.b_out = np.zeros( n_output)
        # second matrix of weights - determines size of a_out (an z_out)
        self.w_out = self.random.normal(loc=0.0,scale=0.1,size=(self.d_hidden, n_output))

        epoch_strlen = len( str( self.n_iter))# for program format
        #self.dEval    = { 'cost' : [], 'acc_tr' : [], 'acc_valid' : []}
        y_train_enc = self.onehot( y_train)
        #-------------------------------------------------------
        #            step B: forward propagation
        #-------------------------------------------------------
        #iterate over training epochs
        for i in range(self.n_iter):
            # iterate over minibatches
            indices = np.arange( X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)

            # forward propagation using minibatch sub-sample
            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                      1, self.minibatch_size):
                # sample n random indices from indices array (shuffled integers )
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                # forward propagation
                z_h, a_h, z_out, a_out = self._forward( X_train[batch_idx])

                # -------------------------------------------------------
                #           step C: back-propagation of errors#######
                # -------------------------------------------------------
                # [n_instance, n_targets], Raschka Eq. 5, page 418
                error_out = a_out - y_train_enc[batch_idx]
                # print( 'activation and OhV')
                # print( a_out, y_train_enc[batch_idx])
                # print( gerger)
                # partial derivative  [n_instance, d_hidden], Raschka Eq. 7, page 419
                sigmoid_deriv_h = a_h * (1. - a_h)
                error_h = (np.dot(error_out, self.w_out.T) * sigmoid_deriv_h)

                # [d_hidden, n_instance] dot [n_instance, n_target] -> [n_hidden, n_target]
                grad_w_out = np.dot( a_h.T, error_out)
                grad_b_out = np.sum( error_out, axis=0)

                # [n_instance, n_target] dot [n_target, d_hidden] -> [n_instance, d_hidden]
                grad_w_h = np.dot( X_train[batch_idx].T, error_h)
                grad_b_h = np.sum( error_h, axis = 0)

                #regularization and weight updates
                self.w_h -= self.eta * (grad_w_h + self.L2*self.w_h)
                self.b_h -= self.eta * grad_b_h

                #output layer weights
                self.w_out -= self.eta * ( grad_w_out + self.L2*self.w_out)
                self.b_out -= self.eta * grad_b_out
            # -------------------------------------------------------
            #           step D: evaluation
            # -------------------------------------------------------
            # test performance after each epoch to check how training progresses
            z_h, a_h, z_out, a_out = self._forward( X_train)
            cost = self._compute_cost(y_enc=y_train_enc,a_out=a_out)
            y_train_pred = self.predict( X_train)
            y_valid_pred = self.predict( X_valid)

            # compare train and validation accuracy at every epoch
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float64) / X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float64) / X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f ' '| Train/Valid Acc.: %.2f%%/%.2f%% '%(
                                epoch_strlen, i + 1, self.n_iter, cost, train_acc * 100, valid_acc * 100))
            sys.stderr.flush()

            # store results in dEval to check training performance
            self.dEval['cost'].append( cost)
            self.dEval['acc_train'].append( train_acc)
            self.dEval['acc_valid'].append( valid_acc)
        return self

    def predict(self, mX):
        """
            - for integer class labels from 0 to n, where n is the total number
              of unique class labels

        :parameter
        ---------------
         mX: array shape = [n_instance, n_feature]
             matrix with original data set
        :return:
        y_hat: array shape = [n_instances]
               predicted class labels
        """
        z_h, a_h, z_out, a_out = self._forward( mX)
        # find unique output unit with largest value of sum of weighted input = y_pred
        y_pred = np.argmax( z_out, axis = 1)
        return y_pred
