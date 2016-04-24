import os
import sys
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams

from treatment import *
from creat import *

rng = numpy.random.RandomState(23456)
srng = MRG_RandomStreams(567432)
batch_size = 100
learning_rate = 0.01
momentum = 0.9

Treat = treatment(batch_size, learning_rate)
n_train_batches, n_valid_batches, n_test_batches = Treat.init_data()

classifier = Network(rng=rng, srng=srng, batch_size=batch_size,
                     architecture=(
                         ("data", (1, 60, 300)),
                         ("reshape", (1, 60, 300)),
                         # ("conv", (300, 3, 300), 0.7, 1),
                         # ("pool", (58, 1), (1, 1), "max"),
                         ("bigcp", (400, 3, 300), (400, 4, 300), (400, 5, 300), (58, 1), (57, 1), (56, 1), 0.7, 1),
                         ("reshape", (1200,)),
                         ("fc", (800,), 0.5, 1, "relu"),
                         ("softmax", (5,)),
                         ("branchout", "empty")
                     ))

print '... training'
patience = 42500
patience_increase = 2
improvement_threshold = 0.995
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

    if epoch < 80:
        learning_rate = 0.1
    elif 80 <= epoch < 500:
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
                    print (('patience now %f,') % (patience))
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
    print (('this batch takes %f,') % (this_batch_end - this_batch_start))

end_time = time.clock()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iterationation %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iteration + 1, test_score * 100.))
print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
