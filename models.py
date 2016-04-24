import numpy
import theano
import theano.tensor as T
import theano.tensor.nnet

from theano.tensor.nnet import conv
from theano.tensor.signal import downsample


class ConvolutionLayer(object):
    def __init__(self, rng, image_shape, filter_shape, drop=1., activation=None):
        # assert filter_shape[1] == image_shape[1]
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:])) / 4
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                             dtype=theano.config.floatX), borrow=True)
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        self.drop = drop
        self.filter_shape = filter_shape
        self.shape_in = image_shape
        self.params = [self.W, self.b]
        self.paramsl2 = [self.W]
        self.activation = activation

    def output(self, input, mask=None):
        if mask is None:

            drop_in = input * self.drop
        else:
            drop_in = input * mask

        conv_out = conv.conv2d(input=drop_in, filters=self.W, filter_shape=self.filter_shape,
                               image_shape=self.shape_in)
        linout = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        output = (
            linout if self.activation is None
            else self.activation(linout)
        )
        return output


class PoolingLayer(object):
    def __init__(self, shape_in, poolsize=(2, 2), drop=1.):
        self.poolsize = poolsize
        self.drop = drop
        self.params = []
        self.paramsl2 = []
        self.shape_in = shape_in

    def output(self, input, mask=None):
        pooled_out = downsample.max_pool_2d(input=input, ds=self.poolsize, ignore_border=True)
        output = pooled_out
        return output


class BigConvPoolLayer(object):
    def __init__(self,
                 rng,
                 image_shape,
                 filter_shape1,
                 filter_shape2,
                 filter_shape3,
                 poolsize1,
                 poolsize2,
                 poolsize3,
                 drop=1.,
                 activation=None,
                 ):
        fan_in1 = numpy.prod(filter_shape1[1:])
        fan_out1 = (filter_shape1[0] * numpy.prod(filter_shape1[2:])) / 4
        W_bound1 = numpy.sqrt(6. / (fan_in1 + fan_out1))
        self.W1 = theano.shared(numpy.asarray(rng.uniform(low=-W_bound1, high=W_bound1, size=filter_shape1),
                                             dtype=theano.config.floatX), borrow=True)

        fan_in2 = numpy.prod(filter_shape2[1:])
        fan_out2 = (filter_shape2[0] * numpy.prod(filter_shape2[2:])) / 4
        W_bound2 = numpy.sqrt(6. / (fan_in2 + fan_out2))
        self.W2 = theano.shared(numpy.asarray(rng.uniform(low=-W_bound2, high=W_bound2, size=filter_shape2),
                                             dtype=theano.config.floatX), borrow=True)

        fan_in3 = numpy.prod(filter_shape3[1:])
        fan_out3 = (filter_shape3[0] * numpy.prod(filter_shape3[2:])) / 4
        W_bound3 = numpy.sqrt(6. / (fan_in3 + fan_out3))
        self.W3 = theano.shared(numpy.asarray(rng.uniform(low=-W_bound3, high=W_bound3, size=filter_shape3),
                                              dtype=theano.config.floatX), borrow=True)

        b_values1 = numpy.zeros((filter_shape1[0],), dtype=theano.config.floatX)
        self.b1 = theano.shared(value=b_values1, borrow=True)
        b_values2 = numpy.zeros((filter_shape2[0],), dtype=theano.config.floatX)
        self.b2 = theano.shared(value=b_values2, borrow=True)
        b_values3 = numpy.zeros((filter_shape3[0],), dtype=theano.config.floatX)
        self.b3 = theano.shared(value=b_values3, borrow=True)

        self.drop = drop
        self.shape_in = image_shape
        self.params = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
        self.paramsl2 = [self.W1, self.W2, self.W3]
        self.activation = activation
        self.filter_shape1 = filter_shape1
        self.filter_shape2 = filter_shape2
        self.filter_shape3 = filter_shape3
        self.poolsize1 = poolsize1
        self.poolsize2 = poolsize2
        self.poolsize3 = poolsize3

    def output(self, input, mask=None):
        if mask is None:
            drop_in = input * self.drop
        else:
            drop_in = input * mask

        conv_out1 = conv.conv2d(input=drop_in, filters=self.W1, filter_shape=self.filter_shape1,
                               image_shape=self.shape_in)
        linout1 = T.nnet.relu(conv_out1 + self.b1.dimshuffle('x', 0, 'x', 'x'))
        output1 = (
            linout1 if self.activation is None
            else self.activation(linout1)
        )
        pooled_out1 = downsample.max_pool_2d(input=output1, ds=self.poolsize1, ignore_border=True)

        conv_out2 = conv.conv2d(input=drop_in, filters=self.W2, filter_shape=self.filter_shape2,
                                image_shape=self.shape_in)
        linout2 = T.nnet.relu(conv_out2 + self.b2.dimshuffle('x', 0, 'x', 'x'))
        output2 = (
            linout2 if self.activation is None
            else self.activation(linout2)
        )
        pooled_out2 = downsample.max_pool_2d(input=output2, ds=self.poolsize2, ignore_border=True)

        conv_out3 = conv.conv2d(input=drop_in, filters=self.W3, filter_shape=self.filter_shape3,
                                image_shape=self.shape_in)
        linout3 = T.nnet.relu(conv_out3 + self.b3.dimshuffle('x', 0, 'x', 'x'))
        output3 = (
            linout3 if self.activation is None
            else self.activation(linout3)
        )
        pooled_out3 = downsample.max_pool_2d(input=output3, ds=self.poolsize3, ignore_border=True)

        output = T.concatenate([pooled_out1, pooled_out2, pooled_out3], axis=1)
        return output


class FullConnectedLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.nnet.relu, drop=1.):
        self.shape_in = input
        self.drop = drop
        self.n_out = int(n_out)

        if W is None:
            W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(n_in + self.n_out)),
                                                 high=numpy.sqrt(6./(n_in + self.n_out)),
                                                 size=(n_in, self.n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.paramsl2 = [self.W]
        self.activation = activation

    def output(self, input, mask=None):
        if mask is None:
            drop_in = input * self.drop
        else:
            drop_in = input * mask

        lin_output = T.dot(drop_in, self.W) + self.b
        output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        return output


class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out, drop=1.):

        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out),
                                                 dtype=theano.config.floatX), name='b', borrow=True)
        self.shape_in = input
        self.drop = drop

        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.paramsl2 = [self.W]

    def output(self, input, mask=None):
        p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        return p_y_given_x


def negative_log_likelihood(x, y):
    return -T.mean(T.log(x)[T.arange(y.shape[0]), y])


def errors(yp, y):
    if y.ndim != yp.ndim:
        raise TypeError('y should have the same shape as self.y_pred',
                        ('y', y.type, 'y_pred', yp.type))
    if y.dtype.startswith('int'):
        return T.mean(T.neq(yp, y))
    else:
        raise NotImplementedError()


def L2_sqr(params):
    l2 = 0
    l2_rate = 5e-5
    for param in params:
        l2 += T.sum(param**2)
    return l2_rate * l2

