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

    with open("D:/theano/SentimentAnalysis/glove.6B.300d.txt", "r") as dic:
        # model modify
        for lined in dic:
            totallined += 1
            print totallined
            dic_parts = lined.strip().split(" ")
            dic_word.append(dic_parts[0])
            dic_vec.append(dic_parts[1:301])

    with open("D:/theano/SentimentAnalysis/train.txt", "r") as f1:
        for line in f1:
            totalline1 += 1
            print totalline1
            line_parts = line.strip().split("\t")
            value = line_parts[0]
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            countword = len(sent_parts)
            newtrain.write(str(value) + "\t" + str(countword) + "\n")
            for word in sent_parts:
                totalword += 1
                wordl = word.lower()
                if wordl in dic_word:
                    ptr = dic_word.index(wordl)
                    newtrain.write(dic_vec[ptr][0])
                    for i in range(1, 300):
                        newtrain.write(" " + str(dic_vec[ptr][i]))
                    newtrain.write("\n")
                else:
                    missword1 += 1
                    print "......" + str(missword1)
                    print "......" + str(word)
                    newtrain.write(str(0))
                    for i in range(1, 300):
                        newtrain.write(" " + str(0))
                    newtrain.write("\n")

    with open("D:/theano/SentimentAnalysis/validation.txt", "r") as f2:
        for line in f2:
            totalline2 += 1
            print totalline2
            line_parts = line.strip().split("\t")
            value = line_parts[0]
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            countword = len(sent_parts)
            newvali.write(str(value) + "\t" + str(countword) + "\n")
            for word in sent_parts:
                totalword += 1
                wordl = word.lower()
                if wordl in dic_word:
                    ptr = dic_word.index(wordl)
                    newvali.write(dic_vec[ptr][0])
                    for i in range(1, 300):
                        newvali.write(" " + str(dic_vec[ptr][i]))
                    newvali.write("\n")
                else:
                    missword2 += 1
                    print "......" + str(missword2)
                    print "......" + str(word)
                    newvali.write(str(0))
                    for i in range(1, 300):
                        newvali.write(" " + str(0))
                    newvali.write("\n")

    with open("D:/theano/SentimentAnalysis/test.txt", "r") as f3:
        for line in f3:
            totalline3 += 1
            print totalline3
            line_parts = line.strip().split("\t")
            value = line_parts[0]
            sent = line_parts[1]
            sent_parts = sent.strip().split(" ")
            countword = len(sent_parts)
            newtest.write(str(value) + "\t" + str(countword) + "\n")
            for word in sent_parts:
                totalword += 1
                wordl = word.lower()
                if wordl in dic_word:
                    ptr = dic_word.index(wordl)
                    newtest.write(dic_vec[ptr][0])
                    for i in range(1, 300):
                        newtest.write(" " + str(dic_vec[ptr][i]))
                    newtest.write("\n")
                else:
                    missword3 += 1
                    print "......" + str(missword3)
                    print "......" + str(word)
                    newtest.write(str(0))
                    for i in range(1, 300):
                        newtest.write(" " + str(0))
                    newtest.write("\n")


