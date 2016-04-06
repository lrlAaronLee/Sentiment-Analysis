import numpy
import theano
import theano.tensor as T
import theano.tensor.nnet

from theano.tensor.nnet import conv


class ConvLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape):
        assert filter_shape[1] == image_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:])) / 4
        W_bound = numpy.sqrt(6./ (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                      dtype=theano.config.floatX), borrow=True)
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]