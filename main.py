import os
import sys
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams

from treatment import *
from creat import *

rng = numpy.random.RandomState(23456)
srng = MRG_RandomStreams(567432)
batch_size = 100
learning_rate = 0.1
momentum = 0.0

Treat = treatment(batch_size)
n_train_batches, n_valid_batches, n_test_batches = Treat.init_data()

classifier = Network(rng=rng, srng=srng, batch_size=batch_size,
                     architecture=(
                         ("data", (60, 1)),
                         ("reshape", (60,)),
                         ("bow", (60, 300)),
                         ("reshape", (1, 60, 300)),
                         # ("conv", (300, 3, 300), 0.7, 1),
                         # ("pool", (58, 1), (1, 1), "max"),
                         ("bigcp", (250, 3, 300), (250, 4, 300), (250, 5, 300), (58, 1), (57, 1), (56, 1), 1, 1),
                         ("reshape", (750,)),
                         ("fc", (750,), 0.5, 1, "relu"),
                         ("softmax", (5,), 0.5),
                         ("branchout", "empty")
                     ))

print '... training'
patience = 85000
patience_increase = 2
improvement_threshold = 0.996
n_epochs = 1000

validation_frequency = n_train_batches  # min(n_train_batches, patience / 2)

best_validation_loss = numpy.inf
best_iteration = 0
test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False
train_model, valid_model, test_model = Treat.build_model(classifier=classifier)

while (epoch < n_epochs) and (not done_looping):
    this_batch_start = time.clock()
    epoch += 1

    if 0 <= epoch < 45:
        learning_rate = 0.1
    elif 45 <= epoch < 450:
        learning_rate = 0.01
    else:
        learning_rate = 0.001

    for minibatch_index in xrange(n_train_batches):

        iteration = (epoch - 1) * n_train_batches + minibatch_index

        if iteration % 85 == 0:
            print 'training @ iteration = ', iteration
        cost_ij = train_model(minibatch_index, learning_rate, momentum)

        if (iteration + 1) % validation_frequency == 0:
            validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f, %% train loss %f' %
                  (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100., cost_ij[0]))

            if this_validation_loss < best_validation_loss:

                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iteration * patience_increase)
                    print (('patience now %i,') % (patience))
                best_validation_loss = this_validation_loss
                best_iteration = iteration
                print "get new minimize!"
                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)
                # test_losses = [test_model(i) for i in xrange(n_test_batches)]
                # test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of '
                          'best model %f %%') %
                     (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

        if patience <= iteration:
            done_looping = True
            break

    this_batch_end = time.clock()
    print (('this batch takes %f,') % (this_batch_end - this_batch_start))

end_time = time.clock()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iterationation %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iteration + 1, test_score * 100.))
print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
