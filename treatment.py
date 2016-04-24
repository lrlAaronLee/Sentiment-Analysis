import numpy as np

from models import *


class treatment(object):
    def __init__(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        return

    def load_data(self, path, size):
        countlen = -1
        dataset_x = np.zeros(size * 60 * 300, np.float)
        dataset_x = dataset_x.reshape((size, 18000))
        dataset_y = np.zeros(size)
        f = open(path, "r")
        while True:
            line = f.readline()
            if line == "":
                break
            countlen += 1
            line_parts = line.strip().split("\t")
            y = float(line_parts[0])
            if 0 <= y <= 0.2:
                dataset_y[countlen] = 0
            elif 0.2 < y <= 0.4:
                dataset_y[countlen] = 1
            elif 0.4 < y <= 0.6:
                dataset_y[countlen] = 2
            elif 0.6 < y <= 0.8:
                dataset_y[countlen] = 3
            else:
                dataset_y[countlen] = 4
            countword = line_parts[1]
            for i in range(0, int(countword)):
                sent = f.readline()
                sent_parts = sent.strip().split(" ")
                for j in range(0, 300):
                    dataset_x[countlen][i * 300 + j] = sent_parts[j]

        shared_x = theano.shared(np.asarray(dataset_x, dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(dataset_y, dtype=theano.config.floatX), borrow=True)
        # print(shared_y.get_value())
        return shared_x, T.cast(shared_y, 'int32')

    def init_data(self):
        rng = np.random.RandomState(23455)

        datasize = [8544, 2210, 1101]
        trainpath = "D:/theano/SentimentAnalysis/train_new.txt"
        validpath = "D:/theano/SentimentAnalysis/validation_new.txt"
        testpath = "D:/theano/SentimentAnalysis/test_new.txt"

        self.sentences_train_x, self.sentences_train_y = self.load_data(trainpath, datasize[0])
        self.sentences_valid_x, self.sentences_valid_y = self.load_data(validpath, datasize[1])
        self.sentences_test_x, self.sentences_test_y = self.load_data(testpath, datasize[2])

        n_train_batches = self.sentences_train_x.get_value(borrow=True).shape[0]
        n_valid_batches = self.sentences_valid_x.get_value(borrow=True).shape[0]
        n_test_batches = self.sentences_test_x.get_value(borrow=True).shape[0]
        n_train_batches /= self.batch_size
        n_valid_batches /= self.batch_size
        n_test_batches /= self.batch_size

        self.index = T.lscalar()
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.learning_rate = T.scalar("learning_rate")
        self.momentum = T.scalar("momentum")
        return n_train_batches, n_valid_batches, n_test_batches

    def train(self, classifier, input, label):
        output = classifier.output(input, mode="train")
        L2 = L2_sqr(classifier.paramsl2)
        costs = negative_log_likelihood(output[-1], label) + L2
        # print "hello"
        # print type(costs)
        # print type(costs[0])
        return [costs]

    def test(self, classifier, input, label):
        output = classifier.output(input, mode="test")
        yp = T.argmax(output[-1], axis=1)
        error = errors(yp, label)
        return [error]

    def build_model(self, classifier):
        print '... building the model'
        index, x, y, batch_size, learning_rate, momentum = \
            self.index, self.x, self.y, self.batch_size, self.learning_rate, self.momentum

        test_model = theano.function(inputs=[index],
                                     outputs=self.test(classifier=classifier, input=x, label=y),
                                     givens={x: self.sentences_test_x[index * batch_size: (index + 1) * batch_size],
                                             y: self.sentences_test_y[index * batch_size: (index + 1) * batch_size]})

        validate_model = theano.function(inputs=[index],
                                         outputs=self.test(classifier=classifier, input=x, label=y),
                                         givens={x: self.sentences_valid_x[index * batch_size: (index + 1) * batch_size],
                                                 y: self.sentences_valid_y[index * batch_size: (index + 1) * batch_size]})

        grads = T.grad(self.train(classifier=classifier, input=x, label=y)[0], classifier.params)

        updates = [(param, param - learning_rate * grads[i] + momentum * classifier.last_updates[i])
                   for i, param in enumerate(classifier.params)] + \
                  [(last_update, - learning_rate * grads[i] + momentum * last_update)
                   for i, last_update in enumerate(classifier.last_updates)]

        train_model = theano.function(inputs=[index, learning_rate, momentum],
                                      outputs=self.train(classifier=classifier, input=x, label=y),
                                      updates=updates,
                                      givens={x: self.sentences_train_x[index * batch_size: (index + 1) * batch_size],
                                              y: self.sentences_train_y[index * batch_size: (index + 1) * batch_size]})

        return train_model, validate_model, test_model
