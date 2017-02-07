import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet.bn import batch_normalization
import copy
import adam
from scipy.stats.mstats import zscore
import cPickle
from warnings import warn

def relu(x): return T.switch(x > 0, x, 0.)

theano.config.floatX = 'float32'

#
# Output layer
#

class MetricLearningLayer(object):

    def __init__(self, numpy_rng, input, n_in, n_out, W_values=None):

        if W_values is None:
            W_values = numpy.asarray(numpy_rng.randn(n_in, n_out) / numpy.sqrt(n_in), dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, name='W', borrow=True)

        self.n_out = n_out  # keep track of output dimensionality

        # input[0] is target; input[1] is atlas

        self.t = T.addbroadcast(input[0], 1)
        self.D = input[1]

        self.diffs = T.dot(self.t - self.D, self.W)

        self.l2n = T.sum(self.diffs ** 2., axis=2)

        # parameters of the model
        self.params = [self.W]

        # keep track of model input
        self.input = input

    def get_l2n(self): return self.l2n  # get ssd

    def get_w(self): return T.nnet.softmax(-self.l2n)  # get label fusion weights

    def lf_cost(self, y):  # label fusion cost

        aux = T.exp(-self.l2n)
        lf = T.sum(y * aux, axis=1) / T.sum(aux, axis=1)
        return -T.mean(T.log(lf))

    def kl_div(self, rho):  # kl divergence for sparsity regularization

        rho_aux = T.addbroadcast(rho, 1)
        aux = T.exp(-self.l2n)
        h = T.mean(aux, axis=0)
        return -T.sum(rho_aux * T.log(h) + (1. - rho_aux) * T.log(1. - h))

    def get_acc(self, y):  # get accuracy

        aux = T.exp(-self.l2n)
        lf = T.sum(y * aux, axis=1) / T.sum(aux, axis=1)
        return T.mean(lf > 0.5)

#
# Hidden layer
#

class HiddenLayer(object):

    def __init__(self, numpy_rng, input, n_in, n_out, bn_flag, activation=None, vflag=None, bn_stats=None, W_values=None, b_values=None, gamma_values=None):

        self.n_out = n_out
        self.bn_flag = bn_flag

        # input[0] is target; input[1] is atlas

        if W_values is None:
            W_values = numpy.asarray(numpy_rng.randn(n_in, n_out) / numpy.sqrt(n_in), dtype=theano.config.floatX)

            if activation == T.nnet.sigmoid: W_values *= 4.

            if activation == T.nnet.relu: W_values /= 2.


        if b_values is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.params = [self.W, self.b]

        t_W = T.dot(input[0], self.W)
        D_W = T.dot(input[1], self.W)

        #
        # Batch normalization

        if bn_flag:

            if gamma_values is None:
                gamma_values = numpy.ones((n_out,), dtype=theano.config.floatX)

            self.gamma = theano.shared(value=gamma_values, name='gamma')

            self.params.append(self.gamma)

            batch_data = T.concatenate([t_W, T.reshape(D_W, (D_W.shape[0] * D_W.shape[1], 1, D_W.shape[2]))], axis=0)

            bn_mu = T.addbroadcast(bn_stats[0], 0); bn_mu = T.addbroadcast(bn_mu, 1)
            bn_std = T.addbroadcast(bn_stats[1], 0); bn_std = T.addbroadcast(bn_std, 1)

            self.bn_mu_switch = T.switch(vflag, bn_mu, batch_data.mean((0,), keepdims=True))
            self.bn_std_switch = T.switch(vflag, bn_std, batch_data.std((0,), keepdims=True))

            lin_t = batch_normalization(inputs=t_W, gamma=self.gamma, beta=self.b, mean=self.bn_mu_switch, std=self.bn_std_switch, mode='low_mem')
            lin_D = batch_normalization(inputs=D_W, gamma=self.gamma, beta=self.b, mean=self.bn_mu_switch, std=self.bn_std_switch, mode='low_mem')

        else:

            lin_t = t_W + self.b
            lin_D = D_W + self.b

        # Layer output

        if activation is not None:
            self.output = (activation(lin_t), activation(lin_D))
        else:
            self.output = (lin_t, lin_D)


    def get_act_stats(self):  # get activation statistics

        act_data = T.concatenate([self.output[0], self.output[1]], axis=1)

        return (act_data.mean(), act_data.std())


    def get_bn_stats(self):  # get batch normalization statistics

        return (self.bn_mu_switch, self.bn_std_switch)


#
# Deep network for metric learning
#

class DeepML(object):

    def __init__(self, numpy_rng, n_in, n_out, n_hidden_units_list, bn_flag, activation_str,
                 labels_list=None, patch_rad=None, patch_norm=None, ds_fact_list=None):

        self.epoch = 0.0

        self.numpy_rng = numpy_rng
        self.activation_str = activation_str
        self.bn_flag = True if bn_flag and n_hidden_units_list else False

        self.labels_list = labels_list
        self.patch_rad = patch_rad
        self.patch_norm = patch_norm
        self.ds_fact_list = ds_fact_list

        self.beta0 = -1.0  # for nlbeta

        #
        # Symbolic variables

        self.t = T.ftensor3("t")  # central patch (sup.)
        self.D = T.ftensor3("D")  # neighboring patches
        self.input = (self.t, self.D)

        self.y = T.fmatrix("y")  # ground truth labels

        self.vflag = T.bscalar("vflag")  # validation flag (batch normalization)
        self.bn_mu_list = [copy.copy(T.ftensor3()) for i in range(len(n_hidden_units_list))]
        self.bn_std_list = [copy.copy(T.ftensor3()) for i in range(len(n_hidden_units_list))]

        self.L1 = T.fscalar()
        self.L2_sqr = T.fscalar()
        self.rho = T.fmatrix()  # for KL divergence

        self.bn_mu_list_values, self.bn_std_list_values = [], []

        # Activation

        activation = None
        if activation_str == 'relu': activation = relu#T.nnet.relu
        if activation_str == 'tanh': activation = T.tanh
        if activation_str == 'sigmoid': activation = T.nnet.sigmoid
        if activation_str == 'none': activation = None

        #
        # build multilayer

        self.hidden_layers_list = []
        next_layer_input = self.input
        next_n_in = n_in
        next_n_out = n_hidden_units_list[0] if n_hidden_units_list else n_out

        for i, (n_units, bn_mu, bn_std) in enumerate(zip(n_hidden_units_list, self.bn_mu_list, self.bn_std_list)):

            hidden_layer = HiddenLayer(numpy_rng=numpy_rng,
                                input=next_layer_input,
                                n_in=next_n_in,
                                n_out=next_n_out,
                                bn_flag=bn_flag,
                                activation=activation,
                                vflag=self.vflag,
                                bn_stats=(bn_mu, bn_std))

            self.hidden_layers_list.append(hidden_layer)

            next_layer_input = hidden_layer.output
            next_n_in = n_hidden_units_list[i]
            next_n_out = n_hidden_units_list[i + 1] if i < len(n_hidden_units_list) - 1 else n_out

        #
        # Metric learning (output) layer

        # numpy_rng, input, n_in, n_out, W_values=None
        self.metric_layer = MetricLearningLayer(numpy_rng=numpy_rng,
                                                input=next_layer_input,
                                                n_in=next_n_in,
                                                n_out=next_n_out)

        # Parameters

        self.updates = []

        self.params = [param for hidden_layer in self.hidden_layers_list for param in hidden_layer.params] + self.metric_layer.params

        #
        # Functions

        self.bn_dummy = [numpy.copy(numpy.zeros((1, 1, hidden_layer.n_out), dtype=theano.config.floatX)) for hidden_layer in self.hidden_layers_list]

        # create givens for theano functions
        self.givens = {}
        self.givens[self.vflag] = numpy.int8(0)
        for i in range(len(self.hidden_layers_list)):
            self.givens[self.bn_mu_list[i]] = self.bn_dummy[i]
            self.givens[self.bn_std_list[i]] = self.bn_dummy[i]

        # functions for obtaining batchnorm statistics (for validation)
        self.fn_bn_stats_list = []
        for hidden_layer in self.hidden_layers_list:
            if hidden_layer.bn_flag:
                self.fn_bn_stats_list.append(
                    theano.function(inputs=[self.t, self.D], outputs=hidden_layer.get_bn_stats(),
                                    givens=self.givens, on_unused_input='ignore'))
            else:
                self.fn_bn_stats_list.append(None)

        self.fn_cost = theano.function(inputs=[self.t, self.D, self.y],
                                       outputs=self.metric_layer.lf_cost(self.y),
                                       givens=self.givens, on_unused_input='ignore')

        self.fn_acc = theano.function(inputs=[self.t, self.D, self.y], outputs=self.metric_layer.get_acc(self.y),
                                      givens=self.givens, on_unused_input='ignore')

        self.fn_l2n = theano.function(inputs=[self.t, self.D], outputs=self.metric_layer.get_l2n(),
                                      givens=self.givens, on_unused_input='ignore')

        self.fn_w = theano.function(inputs=[self.t, self.D], outputs=self.metric_layer.get_w(),
                                    givens=self.givens, on_unused_input='ignore')

        self.fn_kl = theano.function(inputs=[self.t, self.D, self.rho], outputs=self.metric_layer.kl_div(self.rho),
                                     givens=self.givens, on_unused_input='ignore')

        self.fn_w_val = theano.function(inputs=[self.t, self.D] + self.bn_mu_list + self.bn_std_list,
                                        outputs=self.metric_layer.get_w(),
                                        givens={self.vflag: numpy.int8(1)}, on_unused_input='ignore')


    # Regularization variables

        self.L1 = abs(self.metric_layer.W).sum()
        self.L2_sqr = (self.metric_layer.W ** 2).sum()

        for hidden_layer in self.hidden_layers_list:
            self.L1 += abs(hidden_layer.W).sum()
            self.L2_sqr += (hidden_layer.W ** 2).sum()


    # Alternate constructor from filename (to be used by patch-based label fusion)

    @classmethod
    def fromfile(cls, numpy_rng, file_path):

        epoch, activation_str, beta0, labels_list, patch_rad, patch_norm, ds_fact_list, bn_mu_list_values, bn_std_list_values, params_list = read_multilayer(file_path)

        n_in = params_list[0][0].shape[0]
        n_out = params_list[-1][0].shape[1]
        n_hidden_units_list = [param[0].shape[1] for param in params_list[:-1]]
        bn_flag = len(params_list[0]) == 3 if len(params_list) > 1 else False

        embedder = cls(numpy_rng, n_in, n_out, n_hidden_units_list, bn_flag, activation_str, labels_list, patch_rad, patch_norm, ds_fact_list)

        embedder.beta0 = beta0

        if bn_flag:
            embedder.bn_mu_list_values = bn_mu_list_values
            embedder.bn_std_list_values = bn_std_list_values

        for ini_param, hidden_layer in zip(params_list[:-1], embedder.hidden_layers_list):
            hidden_layer.params[0].set_value(numpy.asarray(ini_param[0], dtype=theano.config.floatX))
            hidden_layer.params[1].set_value(numpy.asarray(ini_param[1], dtype=theano.config.floatX))
            if bn_flag:
                hidden_layer.params[2].set_value(numpy.asarray(ini_param[2], dtype=theano.config.floatX))

        embedder.metric_layer.params[0].set_value(numpy.asarray(params_list[-1][0], dtype=theano.config.floatX))

        embedder.epoch = epoch

        return embedder

    # Functions to probe the network

    def get_cost(self, Tp, Ap, Y, use_cost=None):

        return self.fn_cost(numpy.asarray(Tp, dtype=theano.config.floatX),
                            numpy.asarray(Ap, dtype=theano.config.floatX),
                            numpy.asarray(Y, dtype=theano.config.floatX))

    def get_acc(self, Tp, Ap, Y): return self.fn_acc(numpy.asarray(Tp, dtype=theano.config.floatX),
                                                     numpy.asarray(Ap, dtype=theano.config.floatX),
                                                     numpy.asarray(Y, dtype=theano.config.floatX))

    def get_l2n(self, Tp, Ap): return self.fn_l2n(numpy.asarray(Tp, dtype=theano.config.floatX),
                                                  numpy.asarray(Ap, dtype=theano.config.floatX))

    def get_w(self, Tp, Ap): return self.fn_w(numpy.asarray(Tp, dtype=theano.config.floatX),
                                              numpy.asarray(Ap, dtype=theano.config.floatX))

    def get_kl(self, Tp, Ap, rho):

        rho_aux = numpy.asarray(rho, dtype=theano.config.floatX)
        rho_aux.shape = (1, 1)
        return self.fn_kl(numpy.asarray(Tp, dtype=theano.config.floatX),
                          numpy.asarray(Ap, dtype=theano.config.floatX), rho_aux)

    def get_w_val(self, Tp, Ap):

        if self.bn_flag:

            assert self.bn_mu_list_values and self.bn_std_list_values, 'Should update batchnorm statistics before getting validation softmax'

            all_inputs = [numpy.asarray(Tp, dtype=theano.config.floatX),
                          numpy.asarray(Ap, dtype=theano.config.floatX)] + self.bn_mu_list_values + self.bn_std_list_values
            w = self.fn_w_val(*all_inputs)
        else:
            w = self.fn_w(numpy.asarray(Tp, dtype=theano.config.floatX),
                          numpy.asarray(Ap, dtype=theano.config.floatX))

        return w

    # Get list of batchnorm stats (for each layer)

    def update_bn_stats_list(self, Tp, Ap):

        self.bn_mu_list_values, self.bn_std_list_values = [], []
        for fn_bn_stats, bn_dummy in zip(self.fn_bn_stats_list, self.bn_dummy):
            bn_mu, bn_std = fn_bn_stats(numpy.asarray(Tp, dtype=theano.config.floatX),
                                        numpy.asarray(Ap, dtype=theano.config.floatX)) if fn_bn_stats else (bn_dummy, bn_dummy)
            self.bn_mu_list_values.append(numpy.asarray(bn_mu, dtype=theano.config.floatX))
            self.bn_std_list_values.append(numpy.asarray(bn_std, dtype=theano.config.floatX))


    # Train function

    def get_train_function(self, learning_rate, L1_reg, L2_reg, sparse_reg):

        # exponential similarity (with sparsity)
        cost = self.metric_layer.lf_cost(self.y) + \
               numpy.asarray(sparse_reg, dtype=theano.config.floatX) * self.metric_layer.kl_div(self.rho) + \
               numpy.asarray(L1_reg, dtype=theano.config.floatX) * self.L1 + \
               numpy.asarray(L2_reg, dtype=theano.config.floatX) * self.L2_sqr

        # gparams = [T.grad(cost, param) for param in self.params]
        # self.updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]  # vanilla SGD

        self.updates = adam.Adam(cost=cost, params=self.params, lr=learning_rate)  # adam-based SGD

        train_model = theano.function(inputs=[self.t, self.D, self.y, self.rho], outputs=cost, updates=self.updates,
                                      givens=self.givens, on_unused_input='ignore')

        return train_model


    # Write multilayer

    def write_multilayer(self, file_path, Tp_est=None, Ap_est=None, epoch=None):

        f = open(file_path, 'wb')

        # epoch
        cPickle.dump(epoch, f, protocol=cPickle.HIGHEST_PROTOCOL)

        # activation
        cPickle.dump(self.activation_str, f, protocol=cPickle.HIGHEST_PROTOCOL)

        # label fusion parameters
        cPickle.dump(self.beta0, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.labels_list, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.patch_rad, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.patch_norm, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.ds_fact_list, f, protocol=cPickle.HIGHEST_PROTOCOL)

        # bn params
        if self.bn_flag and Tp_est is not None and Ap_est is not None:
            self.update_bn_stats_list(numpy.asarray(Tp_est, dtype=theano.config.floatX),
                                      numpy.asarray(Ap_est, dtype=theano.config.floatX))
        if self.bn_flag and (not self.bn_mu_list_values or not self.bn_std_list_values):
            warn('Saving a model with empty batchnorm stats. Patch data should be provided when saving')
        cPickle.dump(self.bn_mu_list_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.bn_std_list_values, f, protocol=cPickle.HIGHEST_PROTOCOL)

        # multilayer params
        for layer in self.hidden_layers_list:
            params_list = [param.get_value(borrow=True) for param in layer.params]
            cPickle.dump(params_list, f, protocol=cPickle.HIGHEST_PROTOCOL)

        # metric layer params
        params_list = [param.get_value(borrow=True) for param in self.metric_layer.params]
        cPickle.dump(params_list, f, protocol=cPickle.HIGHEST_PROTOCOL)

        f.close()


    # Backup & restore functions in case of bad updates

    def backup_multilayer_params(self):
        self.multilayer_params_backup = [param.get_value() for param in self.params]
        self.multilayer_updates_backup = [pair[0].get_value() for pair in self.updates]

    def restore_multilayer_params(self):
        for i, param in enumerate(self.params): param.set_value(numpy.asarray(self.multilayer_params_backup[i], dtype=theano.config.floatX), borrow=False)
        for i, pair in enumerate(self.updates): pair[0].set_value(numpy.asarray(self.multilayer_updates_backup[i], dtype=theano.config.floatX), borrow=False)


    # Rescale metric layer weights

    def rescale_metric_layer_weights(self, beta0):
        self.metric_layer.W.set_value(numpy.asarray(self.metric_layer.W.get_value() * numpy.asarray(beta0, dtype=theano.config.floatX), dtype=theano.config.floatX))

    def get_epoch(self): return self.epoch

    def set_epoch(self, epoch): self.epoch = epoch



#
# Id network (for efficient computations)
#


# This layer does not perform any transformation
class IdLayer(object):

    def __init__(self, input):

        # input[0] is target; input[1] is atlas

        self.t = T.addbroadcast(input[0], 1)
        self.D = input[1]

        self.diffs = self.t - self.D

        self.l2n = T.sum(self.diffs ** 2., axis=2)


    def get_l2n(self): return self.l2n

    def get_w(self, beta0):

        beta_aux = T.addbroadcast(beta0, 0)
        beta_aux = T.addbroadcast(beta_aux, 1)

        return T.nnet.softmax(-beta_aux * self.l2n)

    def lf_cost(self, y, beta0):

        beta_aux = T.addbroadcast(beta0, 0)
        beta_aux = T.addbroadcast(beta_aux, 1)

        aux = T.exp(-beta_aux * self.l2n)
        lf = T.sum(y * aux, axis=1) / T.sum(aux, axis=1)
        return -T.mean(T.log(lf))

    def kl_div(self, beta0, rho):

        beta_aux = T.addbroadcast(beta0, 0)
        beta_aux = T.addbroadcast(beta_aux, 1)
        rho_aux = T.addbroadcast(rho, 1)
        # rho_aux = T.addbroadcast(rho_aux, 1)

        aux = T.exp(-beta_aux * self.l2n)
        h = T.mean(aux, axis=0)
        return -T.sum(rho_aux * T.log(h) + (1. - rho_aux) * T.log(1. - h))

    def get_acc(self, y, beta0):

        beta_aux = T.addbroadcast(beta0, 0)
        beta_aux = T.addbroadcast(beta_aux, 1)

        aux = T.exp(-beta_aux * self.l2n)
        lf = T.sum(y * aux, axis=1) / T.sum(aux, axis=1)
        return T.mean(lf > 0.5)


class IdNet(object):

    def __init__(self):

        #
        # Symbolic variables

        self.t = T.ftensor3("t")  # central patch (sup.)
        self.D = T.ftensor3("D")  # neighboring patches
        self.input = (self.t, self.D)

        self.y = T.fmatrix("y")  # ground truth labels

        self.beta0 = T.fmatrix()
        self.rho = T.fmatrix()

        #
        # Identity layer

        self.id_layer = IdLayer(input=self.input)

        # functions

        self.fn_cost = theano.function(inputs=[self.t, self.D, self.y, self.beta0], outputs=self.id_layer.lf_cost(self.y, self.beta0))

        self.fn_l2n = theano.function(inputs=[self.t, self.D], outputs=self.id_layer.get_l2n())

        self.fn_w = theano.function(inputs=[self.t, self.D, self.beta0], outputs=self.id_layer.get_w(self.beta0))

        self.fn_kl = theano.function(inputs=[self.t, self.D, self.beta0, self.rho], outputs=self.id_layer.kl_div(self.beta0, self.rho))

        self.fn_acc = theano.function(inputs=[self.t, self.D, self.y, self.beta0], outputs=self.id_layer.get_acc(self.y, self.beta0))


    # Functions to probe the network

    def get_cost(self, Tp, Ap, Y, beta0, use_cost=None):
        beta_aux = numpy.asarray(beta0, dtype=theano.config.floatX)
        beta_aux.shape = (1, 1)
        return self.fn_cost(numpy.asarray(Tp, dtype=theano.config.floatX),
                            numpy.asarray(Ap, dtype=theano.config.floatX),
                            numpy.asarray(Y, dtype=theano.config.floatX),
                            beta_aux)

    def get_acc(self, Tp, Ap, Y, beta0):
        beta_aux = numpy.asarray(beta0, dtype=theano.config.floatX)
        beta_aux.shape = (1, 1)
        return self.fn_acc(numpy.asarray(Tp, dtype=theano.config.floatX),
                            numpy.asarray(Ap, dtype=theano.config.floatX),
                            numpy.asarray(Y, dtype=theano.config.floatX),
                            beta_aux)

    def get_l2n(self, Tp, Ap):
        return self.fn_l2n(numpy.asarray(Tp, dtype=theano.config.floatX),
                            numpy.asarray(Ap, dtype=theano.config.floatX))

    def get_w(self, Tp, Ap, beta0):
        beta_aux = numpy.asarray(beta0, dtype=theano.config.floatX)
        beta_aux.shape = (1, 1)
        return self.fn_w(numpy.asarray(Tp, dtype=theano.config.floatX),
                         numpy.asarray(Ap, dtype=theano.config.floatX),
                         beta_aux)

    def get_kl(self, Tp, Ap, beta0, rho):
        beta_aux = numpy.asarray(beta0, dtype=theano.config.floatX)
        beta_aux.shape = (1, 1)
        rho_aux = numpy.asarray(rho, dtype=theano.config.floatX)
        rho_aux.shape = (1, 1)
        return self.fn_kl(numpy.asarray(Tp, dtype=theano.config.floatX),
                          numpy.asarray(Ap, dtype=theano.config.floatX),
                          beta_aux, rho_aux)


#
# Auxiliary functions
#

def read_multilayer(file_path):

    f = open(file_path, 'rb')

    epoch = cPickle.load(f)
    activation_str = cPickle.load(f)

    # label fusion parameters
    beta0 = cPickle.load(f)
    labels_list = cPickle.load(f)
    patch_rad = cPickle.load(f)
    patch_norm = cPickle.load(f)
    ds_fact_list = cPickle.load(f)

    bn_mu_list_values = cPickle.load(f)
    bn_std_list_values = cPickle.load(f)

    params_list = []
    while True:
        try:
            params_list.append(cPickle.load(f))
        except:
            break
    f.close()

    return (epoch, activation_str, beta0, labels_list, patch_rad, patch_norm, ds_fact_list, bn_mu_list_values, bn_std_list_values, params_list)



def normalize_patches(patch_norm, Patches):
    Patches_out = numpy.asarray([])
    if patch_norm == 'zscore': Patches_out = zscore(numpy.asarray(Patches, dtype=theano.config.floatX), axis=2)
    elif patch_norm == 'l2': Patches_out = numpy.asarray(Patches, dtype=theano.config.floatX) / \
                                           numpy.linalg.norm(numpy.asarray(Patches, dtype=theano.config.floatX), ord=2, axis=2)[..., numpy.newaxis]
    Patches_out[numpy.logical_not(numpy.isfinite(Patches_out))] = 0.
    return Patches_out



