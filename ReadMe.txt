extractor.py:对句子进行简单的处理，生成三个文件分别存储训练集、验证集和测试集的句子
getm.py:生成三个文件存储用数字向量表示的训练集、验证集和测试集，生成词向量index
main.py:主函数，定义网络以及模型训练部分
models.py:定义网络所需的各种层的类
creat.py:根据定义的网络及models.py中层的定义生成网络
treatment.py:读入数据及确定网络中数据之间的函数关系
词向量： Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip 
        http://nlp.stanford.edu/projects/glove/ 
