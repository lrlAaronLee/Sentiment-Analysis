# -*- coding: UTF-8 -*-

import os
import sys
import time
import numpy
import theano
import theano.tensor as T
import theano.tensor.nnet as tn

from load_data import load_data
from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression

if __name__ == "__main__":
    print(theano.config.device)

    learning_rate = 0.01
    n_epochs = 100
    nkerns = [200, 200, 200]
    batch_size = 200

    finfo = open('finfo.txt', 'w')
    rng = numpy.random.RandomState(23455)

    datasize = [8544, 2210, 1101]
    bigsize = [8600, 2400, 1200]
    trainpath = "D:/theano/SentimentAnalysis/train.txt"
    validpath = "D:/theano/SentimentAnalysis/validation.txt"
    testpath = "D:/theano/SentimentAnalysis/test.txt"

    sentences_train_x, sentences_train_y = load_data(trainpath, bigsize[0])
    sentences_valid_x, sentences_valid_y = load_data(validpath, bigsize[1])
    sentences_test_x, sentences_test_y = load_data(testpath, bigsize[2])

    n_train_batches = sentences_train_x.get_value(borrow=True).shape[0]
    n_valid_batches = sentences_valid_x.get_value(borrow=True).shape[0]
    n_test_batches = sentences_test_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    print '... building the model'
    print >> finfo, '... building the model'

    layer0_input = x.reshape((batch_size, 1, 60, 40))

    layer1_1 = ConvLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 60, 40), filter_shape=(nkerns[0], 1, 3, 40))
    layer1_2 = ConvLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 60, 40), filter_shape=(nkerns[1], 1, 4, 40))
    layer1_3 = ConvLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 60, 40), filter_shape=(nkerns[2], 1, 5, 40))

    layer2_1 = PoolLayer(input=layer1_1.output, poolsize=(58, 1))
    layer2_2 = PoolLayer(input=layer1_2.output, poolsize=(57, 1))
    layer2_3 = PoolLayer(input=layer1_3.output, poolsize=(56, 1))

    l2_1 = layer2_1.output.flatten()
    l2_2 = layer2_2.output.flatten()
    l2_3 = layer2_3.output.flatten()
    layer3_input = l2_1 + l2_2 + l2_3  # ?????????

    layer3 = HiddenLayer(rng, input=layer3_input, n_in=nkerns[1] * 4 * 4, n_out=500, activation=T.nnet.r)

    # ???? 2016.03.29
    layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=5)

    cost = layer4.negative_log_likelihood(y)
    print type(sentences_train_x)
    print type(sentences_test_y)

    test_model = theano.function([index], layer4.errors(y),
                                 givens={x: sentences_test_x[index * batch_size: (index + 1) * batch_size],
                                         y: sentences_test_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer4.errors(y),
                                     givens={x: sentences_valid_x[index * batch_size: (index + 1) * batch_size],
                                             y: sentences_valid_y[index * batch_size: (index + 1) * batch_size]})

    params = layer1_1.params + layer1_2.params + layer1_3.params + layer2_1.params + layer2_2.params + layer2_3.params + layer3.params + layer4.params

    grads = T.grad(cost, params)

    updates = [(param_i, param_i - learning_rate * grad_i)
               for param_i, grad_i in zip(params, grads)]

    train_model = theano.function([index], cost, updates=updates,
                                  givens={x: sentences_train_x[index * batch_size: (index + 1) * batch_size],
                                          y: sentences_train_y[index * batch_size: (index + 1) * batch_size]})

    print '... training'

    print >> finfo, '... training'
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iteration = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        this_batch_start = time.clock()
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            iteration = (epoch - 1) * n_train_batches + minibatch_index
            if iteration % 100 == 0:
                print 'training @ iteration = ', iteration
                print >> finfo, 'training @ iteration = ', iteration
            cost_ij = train_model(minibatch_index)
            if (iteration + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))
                print >> finfo, ('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iteration * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iteration = iteration

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iteration:
                done_looping = True
                break

        this_batch_end = time.clock()
        print('this batch takes ', (this_batch_end - this_batch_start), 's')
        print >> finfo, ('this batch takes %.2fm', (this_batch_end - this_batch_start))

    end_time = time.clock()
    print('Optimization complete.')
    print >> finfo, ('Optimization complete.')
    print('Best validation score of %f %% obtained at iterationation %i, '
        'with test performance %f %%' %
        (best_validation_loss * 100., best_iteration + 1, test_score * 100.))
    print >> finfo, ('Best validation score of %f %% obtained at iterationation %i, '
        'with test performance %f %%' %
        (best_validation_loss * 100., best_iteration + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print >> finfo, (('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
    finfo.close


