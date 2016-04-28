from models import *


def layer_define(rng, batch_size, architecture):
    shape_data = None
    shape_additional_data = None
    layer_list = []
    initial_output = []
    empty_output = []
    empty_peephole = []
    activation_dict = {
        'none': None,
        'relu': T.nnet.relu,
        'sigmoid': T.nnet.sigmoid,
        'tanh': T.tanh,
        'hard': T.nnet.hard_sigmoid,
        'softmax': T.nnet.softmax
    }
    for layer in architecture:
        if layer[0] == 'data':
            shape_data = (batch_size,) + layer[1]
            continue
        elif layer[0] == 'reshape':
            shape_data = (batch_size,) + layer[1]
            continue
        elif layer[0] == 'bow':
            shape_trans = layer[1]
            layer_list.append(GetBoW(shape_data, shape_trans, "D:/theano/SentimentAnalysis/dic_vec.pkl"))
            shape_data = (batch_size,) + shape_trans
        elif layer[0] == 'conv':
            shape_trans = layer[1]
            p_drop = layer[2]
            repeat = layer[3]
            for i in range(repeat):
                layer_list.append(
                    ConvolutionLayer(rng, shape_data, shape_trans[:1]+(shape_data[1],)+shape_trans[1:], p_drop,
                                     activation=T.nnet.relu)
                )
                shape_data = (shape_data[0], shape_trans[0],
                              shape_data[2]+1-shape_trans[1], shape_data[3]+1-shape_trans[2])
            continue
        elif layer[0] == 'pool':
            shape_pool = layer[1]
            st = layer[2]
            layer_list.append(
                PoolingLayer(shape_data, shape_pool)
            )
            shape_data = (shape_data[0], shape_data[1], 1+(shape_data[2]-shape_pool[0])/st[0],
                          1+(shape_data[3]-shape_pool[1])/st[1])
            continue
        elif layer[0] == 'bigcp':
            shape_trans1 = layer[1]
            shape_trans2 = layer[2]
            shape_trans3 = layer[3]
            shape_pool1 = layer[4]
            shape_pool2 = layer[5]
            shape_pool3 = layer[6]
            p_drop = layer[7]
            layer_list.append(
                BigConvPoolLayer(rng, shape_data, shape_trans1[:1]+(shape_data[1],)+shape_trans1[1:],
                                 shape_trans2[:1] + (shape_data[1],) + shape_trans2[1:],
                                 shape_trans3[:1] + (shape_data[1],) + shape_trans3[1:],
                                 shape_pool1, shape_pool2, shape_pool3,
                                 p_drop, activation=T.nnet.relu
                                 )
            )
            shape_data = (shape_data[0], 3*300)
            continue
        elif layer[0] == 'joint':
            if shape_additional_data[2] != shape_data[2] or shape_additional_data[3] != shape_data[3]:
                print("!! shape mismatch")
            shape_data = (shape_data[0], shape_data[1]+shape_additional_data[1], shape_data[2], shape_data[3])
            continue
        elif layer[0] == 'fc':
            shape_trans = layer[1]
            p_drop = layer[2]
            repeat = layer[3]
            activation = activation_dict[layer[4]]
            for i in range(repeat):
                layer_list.append(
                    FullConnectedLayer(rng, shape_data, shape_data[1], shape_trans[0],
                                       activation=activation, drop=p_drop)
                )
                shape_data = (shape_data[0], shape_trans[0])
            continue
        elif layer[0] == 'softmax':
            shape_trans = layer[1]
            p_drop = layer[2]
            layer_list.append(
                SoftmaxLayer(shape_data, shape_data[1], shape_trans[0], p_drop)
            )
            shape_data = (shape_data[0], shape_trans[0])
            continue
        elif layer[0] == 'branchout':
            btype = layer[1]
            if btype == 'initial':
                initial_value = numpy.zeros(shape=shape_data, dtype=theano.config.floatX)
                initial_variable = theano.shared(value=initial_value, name='initial_variable', borrow=True)
                initial_output.append(initial_variable)
            elif btype == 'empty':
                empty_output.append(shape_data)
            else:
                print('unknown symbol !!!\n%s' % layer[1])
            continue
        elif layer[0] == 'branchin':
            shape_additional_data = (batch_size,) + layer[1]
            continue
        elif layer[0] == 'peephole':
            empty_peephole.append(shape_data)
            continue
        else:
            print('unknown symbol !!!\n%s' % layer[0])
    return layer_list, initial_output, empty_output, empty_peephole


def layer_operate(inputs, memories, architecture, layers, batch_size, masks, mode):
    outputs = []
    update_memories = range(len(memories))
    peepholes = []
    data = None
    additional_data = None
    layer_pointer = 0
    for layer in architecture:
        if layer[0] == 'data':
            data = inputs[-1]
            continue
        elif layer[0] == 'branchin':
            input_pointer = layer[2]
            additional_data = inputs[input_pointer]
            continue
        elif layer[0] == 'reshape':
            shape = layer[1]
            data = data.reshape((batch_size,)+shape)
            continue
        elif layer[0] == 'bow':
            data = layers[layer_pointer].output(input=data)
            layer_pointer += 1
        elif layer[0] == 'conv':
            repeat = layer[3]
            for i in range(repeat):
                data = layers[layer_pointer].output(input=data, mask=masks[layers[layer_pointer]])
                layer_pointer += 1
            continue
        elif layer[0] == 'pool':
            data = layers[layer_pointer].output(input=data, mask=masks[layers[layer_pointer]])
            layer_pointer += 1
            continue
        elif layer[0] == 'bigcp':
            data = layers[layer_pointer].output(input=data, mask=masks[layers[layer_pointer]])
            layer_pointer += 1
            continue
        elif layer[0] == 'joint':
            data = T.concatenate([data, additional_data], axis=1)
            continue
        elif layer[0] == 'fc':
            repeat = layer[3]
            for i in range(repeat):
                data = layers[layer_pointer].output(input=data, mask=masks[layers[layer_pointer]])
                layer_pointer += 1
            continue
        elif layer[0] == 'softmax':
            data = layers[layer_pointer].output(input=data, mask=masks[layers[layer_pointer]])
            layer_pointer += 1
            continue
        elif layer[0] == 'branchout':
            outputs.append(data)
            continue
        elif layer[0] == 'peephole':
            if mode == 'observe':
                peepholes.append(data)
            continue
        else:
            print('unknown symbol !!!\n%s' % layer[0])
    return outputs + peepholes, update_memories


class Network(object):
    def __init__(self, rng, srng, batch_size, architecture):
        self.batch_size = batch_size
        self.architecture = architecture
        self.rng = rng
        self.srng = srng

        self.layers, self.initial_output, self.empty_output, self.empty_peephole \
            = layer_define(rng=self.rng, batch_size=self.batch_size, architecture=self.architecture)
        self.params = [param for layer in self.layers for param in layer.params]
        self.paramsl2 = [param for layer in self.layers for param in layer.paramsl2]
        self.last_updates = [theano.shared(value=numpy.zeros(shape=param.get_value().shape, dtype=theano.config.floatX))
                             for param in self.params]
        return

    def output(
            self, input, mode
    ):
        if mode == 'train':
            masks = {layer: self.srng.binomial(size=layer.shape_in, p=layer.drop, dtype='float32') for layer in
                     self.layers}
        else:
            masks = {layer: None for layer in self.layers}
        outputs, memories = layer_operate(inputs=[input], memories=[], architecture=self.architecture,
                                          layers=self.layers, batch_size=self.batch_size, masks=masks, mode=mode)
        return outputs

