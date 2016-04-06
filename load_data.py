import numpy as np
import theano
import theano.tensor as T


def load_data(path, size):
    countlen = -1
    dataset_x = np.zeros(size * 60 * 40, np.int32)
    dataset_x = dataset_x.reshape((size, 2400))
    dataset_y = np.zeros(size)
    with open(path, "r") as f:
        for line in f:
            countlen += 1
            line_parts = line.strip().split("\t")
            y = float(line_parts[0])
            if 0 <= y <= 0.2:
                dataset_y[countlen] = 1
            elif 0.2 < y <= 0.4:
                dataset_y[countlen] = 2
            elif 0.4 < y <= 0.6:
                dataset_y[countlen] = 3
            elif 0.6 < y <= 0.8:
                dataset_y[countlen] = 4
            else:
                dataset_y[countlen] = 5
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            countword = -1
            for word in sent_parts:
                wordlen = len(word)
                countword += 1
                numword = map(ord, word)
                for i in range(0, wordlen):
                    dataset_x[countlen][countword * 40 + i] = numword[i]

    shared_x = theano.shared(np.asarray(dataset_x, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(dataset_y, dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, 'int32')