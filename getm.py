import cPickle
import numpy
import theano

if __name__ == '__main__':
    newtrain = open("train_new.txt", "w")
    newtest = open("test_new.txt", "w")
    newvali = open("validation_new.txt", "w")
    totalword = 0
    missword1 = 0
    missword2 = 0
    missword3= 0
    totalline1 = 0
    totalline2 = 0
    totalline3 = 0
    totallined = 0
    dic_word = []
    dic_vec = []
    ha_word = []

    with open("D:/theano/SentimentAnalysis/train.txt", "r") as f1:
        for line in f1:
            line_parts = line.strip().split("\t")
            value = line_parts[0]
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            for word in sent_parts:
                ha_word.append(word.lower())

    with open("D:/theano/SentimentAnalysis/validation.txt", "r") as f1:
        for line in f1:
            line_parts = line.strip().split("\t")
            value = line_parts[0]
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            for word in sent_parts:
                ha_word.append(word.lower())

    with open("D:/theano/SentimentAnalysis/test.txt", "r") as f1:
        for line in f1:
            line_parts = line.strip().split("\t")
            value = line_parts[0]
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            for word in sent_parts:
                ha_word.append(word.lower())

    ha_word = list(set(ha_word))
    word_len = len(ha_word)

    with open("D:/theano/SentimentAnalysis/glove.840B.300d.txt", "r") as dic:
        for lined in dic:
            totallined += 1
            if totallined % 10000 == 0:
                print totallined
            dic_parts = lined.strip().split(" ")
            if dic_parts[0] in ha_word:
                dic_word.append(dic_parts[0])
                dic_vec.append(dic_parts[1:301])
    nu = []
    for i in range(0, 300):
        nu.append(0)
    dic_vec.append(nu[0:300])
    len_vec = len(dic_vec)
    dic_vecn = numpy.asarray(dic_vec, dtype=theano.config.floatX)

    cPickle.dump(dic_vecn, open("dic_vec.pkl", "wb"))

    with open("D:/theano/SentimentAnalysis/train.txt", "r") as f1:
        for line in f1:
            totalline1 += 1
            print totalline1
            line_parts = line.strip().split("\t")
            value = line_parts[0]
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            newtrain.write(str(value))
            countword = 0
            for word in sent_parts:
                totalword += 1
                wordl = word.lower()
                if wordl == " ":
                    continue
                elif wordl in dic_word:
                    countword += 1
                    ptr = dic_word.index(wordl)
                    newtrain.write(" " + str(ptr))
                else:
                    countword += 1
                    missword1 += 1
                    print "......" + str(missword1)
                    print "......" + str(word)
                    newtrain.write(" " + str(len_vec-1))
            for i in range(countword, 60):
                newtrain.write(" " + str(len_vec-1))
            newtrain.write("\n")

    with open("D:/theano/SentimentAnalysis/validation.txt", "r") as f2:
        for line in f2:
            totalline2 += 1
            print totalline2
            line_parts = line.strip().split("\t")
            value = line_parts[0]
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            countword = 0
            newvali.write(str(value))
            for word in sent_parts:
                totalword += 1
                wordl = word.lower()
                if wordl == " ":
                    continue
                elif wordl in dic_word:
                    countword += 1
                    ptr = dic_word.index(wordl)
                    newvali.write(" " + str(ptr))
                else:
                    countword += 1
                    missword2 += 1
                    print "......" + str(missword2)
                    print "......" + str(word)
                    newvali.write(" " + str(len_vec-1))
            for i in range(countword, 60):
                newvali.write(" " + str(len_vec-1))
            newvali.write("\n")

    with open("D:/theano/SentimentAnalysis/test.txt", "r") as f3:
        for line in f3:
            totalline3 += 1
            print totalline3
            line_parts = line.strip().split("\t")
            value = line_parts[0]
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            countword = 0
            newtest.write(str(value))
            for word in sent_parts:
                totalword += 1
                wordl = word.lower()
                if wordl == " ":
                    continue
                elif wordl in dic_word:
                    countword += 1
                    ptr = dic_word.index(wordl)
                    newtest.write(" " + str(ptr))
                else:
                    countword += 1
                    missword3 += 1
                    print "......" + str(missword3)
                    print "......" + str(word)
                    newtest.write(" " + str(len_vec-1))
            for i in range(countword, 60):
                newtest.write(" " + str(len_vec-1))
            newtest.write("\n")

    print missword1
    print missword2
    print missword3
    print totalword



