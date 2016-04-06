from theano.tensor.signal import downsample


class PoolLayer(object):
    def __init__(self, input, poolsize=(2, 2)):
        self.input = input
        pooled_out = downsample.max_pool_2d(input=input, ds=poolsize, ignore_border=True)
        self.output = pooled_out
