# -*- coding: UTF-8 -*-
import sys


def extractor():
    sentence_type = "full"
    full = "full"
    all = "all"
    data_path = "D:/theano/SentimentAnalysis/"

    dataset_fixes = {"\x83\xc2": "", "-LRB-":"(", "-RRB-":")", "\xc3\x82\xc2\xa0":" "}

    #read dataset split
    dataset_split = {}
    with open(data_path + "datasetSplit.txt", "r") as f:
        next(f)
        for line in f:
            line_parts = line.strip().split(",")
            dataset_split[line_parts[0].strip()] = line_parts[1].strip()

    # read relevant sentences
    sentences1 = []
    sentences2 = []
    sentences3 = []
    with open(data_path + "datasetSentences.txt", "r") as f:
        next(f)
        for line in f:
            line_parts = line.strip().split("\t")
            if len(line_parts) != 2:
                raise ValueError("Unexpected file format")
            if int(dataset_split[line_parts[0]]) == 1:
                sentence = line_parts[1]
                for fix in dataset_fixes:
                    sentence = sentence.replace(fix, dataset_fixes[fix])
                sentences1.append(sentence)
            if int(dataset_split[line_parts[0]]) == 2:
                sentence = line_parts[1]
                for fix in dataset_fixes:
                    sentence = sentence.replace(fix, dataset_fixes[fix])
                sentences2.append(sentence)
            if int(dataset_split[line_parts[0]]) == 3:
                sentence = line_parts[1]
                for fix in dataset_fixes:
                    sentence = sentence.replace(fix, dataset_fixes[fix])
                sentences3.append(sentence)


    # read sentiment labels
    sentiment_labels = {}
    with open(data_path + "sentiment_labels.txt", "r") as f:
        next(f)
        for line in f:
            line_parts = line.strip().split("|")
            if len(line_parts) != 2:
                raise ValueError("Unexpected file format")
            sentiment_labels[line_parts[0]] = float(line_parts[1])

    # read the phrases
    phrases = {}
    with open(data_path + "dictionary.txt", "r") as f:
        for line in f:
            line_parts = line.strip().split("|")
            if len(line_parts) != 2:
                raise ValueError("Unexpected file format")
            phrases[line_parts[0]] = sentiment_labels[line_parts[1]]

    # print the labels and sentences/phrases
    if sentence_type == full:
        f1 = open('train.txt', 'w')
        f2 = open('validation.txt', 'w')
        f3 = open('test.txt', 'w')
        for sentence in sentences1:
            f1.write(str(phrases[sentence]) + "\t" + sentence + "\n")
        for sentence in sentences2:
            f2.write(str(phrases[sentence]) + "\t" + sentence + "\n")
        for sentence in sentences3:
            f3.write(str(phrases[sentence]) + "\t" + sentence + "\n")
        f1.close()
        f2.close()
        f3.close()
    # elif sentence_type == all:
    #    for phrase in phrases:
    #        print_phrase = False
    #        for sentence in sentences:
    #            if sentence.find(phrase) >= 0:
    #                print_phrase = True
    #                break
    #        if print_phrase:
    #            print str(phrases[phrase]) + "\t" + phrase


extractor()
