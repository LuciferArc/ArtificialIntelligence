# -*- coding: utf-8 -*-
import re
import jieba.posseg as pseg
import pandas as pd
import math
import numpy as np
# 加载常用停用词
stopwords1 = [line.rstrip() for line in open('./词库1.txt', 'r', encoding='utf-8')]
stopwords2 = [line.rstrip() for line in open('./词库2.txt', 'r', encoding='utf-8')]
stopwords3 = [line.rstrip() for line in open('./词库3.txt', 'r', encoding='utf-8')]
stopwords4 = [line.rstrip() for line in open('./词库4.txt', 'r', encoding='utf-8')]
stopwords5 = [line.rstrip() for line in open('./词库5.txt', 'r', encoding='utf-8')]
stopwords6 =  [line.rstrip() for line in open('./词库6.txt', 'r', encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4 + stopwords5 + stopwords6
def proc_text(raw_line):
    """
        处理每行的文本数据
        返回分词结果
    """
    # 1. 使用正则表达式去除非中文字符
    #    在 [] 内使用 ^ 表示非，否则表示行首
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    # 将所有非中文字符替换为""
    chinese_only = filter_pattern.sub('', raw_line)

    # 2. 结巴分词+词性标注
    # 返回分词结果列表，包含单词和词性
    words_lst = pseg.cut(chinese_only)

    # 3. 去除停用词
    # 将所有非停用词的词语存到列表里
    meaninful_words = []
    for word, flag in words_lst:
        #if (word not in stopwords) and (flag == 'v'):
            # 也可根据词性去除非动词等
        if word not in stopwords:
            meaninful_words.append(word)

    # 返回一个字符串
    return ' '.join(meaninful_words)


def split_train_test(text_df, size=0.8):
    """
        分割训练集和测试集
        size = 0.8
        表示按二八法则分隔数据集，80%做为训练集，20%测试集
    """
    # 为保证每个类中的数据能在训练集中和测试集中的比例相同，所以需要依次对每个类进行处理
    train_text_df = pd.DataFrame()
    test_text_df = pd.DataFrame()

    # 表示情感值
    labels = [0, 1, 2, 3]
    for label in labels:
        # 找出label的记录
        text_df_w_label = text_df[text_df['label'] == label]
        # 重新设置索引，保证每个类的记录是从0开始索引，方便之后的拆分
        text_df_w_label = text_df_w_label.reset_index()

        # 默认按80%训练集，20%测试集分割
        # 这里为了简化操作，取前80%放到训练集中，后20%放到测试集中
        # 当然也可以随机拆分80%，20%（尝试实现下DataFrame中的随机拆分）

        # 该类数据的行数
        n_lines = text_df_w_label.shape[0]
        # 16432 * 0.8 = 8000
        # 根据size值，获取训练集的行数 math.floor() 求浮点数向下最接近的整数
        split_line_no = math.floor(n_lines * size)

        # 取出当前类的文本的 开始 ~split_line_no 部分行做为训练集
        text_df_w_label_train = text_df_w_label.iloc[:split_line_no, :]
        # 取出当前类的文本的 split_line_no ~ 最后 部分行做为测试集
        text_df_w_label_test = text_df_w_label.iloc[split_line_no:, :]

        # 放入整体训练集，测试集中
        train_text_df = train_text_df.append(text_df_w_label_train)
        test_text_df = test_text_df.append(text_df_w_label_test)

    # 重置索引
    train_text_df = train_text_df.reset_index()
    # 重置索引
    test_text_df = test_text_df.reset_index()
    # 包含所有的训练集 和 测试集
    return train_text_df, test_text_df


def get_word_list_from_data(text_df):
    """
        将数据集中的单词放入到一个列表中
    """
    word_list = []
    # text_df.iterrows()返回一个列表，包含了所有数据的系你想
    # [(行号，内容), (行号，内容), (行号，内容), (行号，内容)......]
    for i, r_data in text_df.iterrows():
        word_list += r_data['text'].split(' ')
    # 包含数据集里所有词语的列表
    return word_list


def extract_feat_from_data(text_df, text_collection, common_words_freqs):
    """
        特征提取
    """
    # 这里只选择TF-IDF特征作为例子
    # 可考虑使用词频或其他文本特征作为额外的特征

    # 取出训练数据集的行数
    n_sample = text_df.shape[0]
    # 取出词频统计的个数
    n_feat = len(common_words_freqs)
    # 取出词频统计的200个词语
    # word表示词语，_表示词频值，返回“所有常用单词”的列表
    common_words = [word for word, _ in common_words_freqs]

    # 向量初始化
    # 构建一个 n_sample行 200列的二维数组，用来保存200个单词在每一行的 tf-idf值
    X = np.zeros([n_sample, n_feat])
    #
    y = np.zeros(n_sample)

    print('提取特征...')
    # i 表示行索引， r_data 表示行数据
    # 每次循环一行
    for i, r_data in text_df.iterrows():
        #每隔5000行就打印一次log
        if (i + 1) % 10000 == 0:
            print('已完成{}个样本的特征提取'.format(i + 1))

        # 每次循环当前行的text文本
        text = r_data['text']

        feat_vec = []
        # 循环遍历常用单词列表
        for word in common_words:
            # 如果word在当前行text文本里，则计算TF-IDF值
            if word in text:
                # 如果在高频词中，计算TF-IDF值
                tf_idf_val = text_collection.tf_idf(word, text)
            else:
                tf_idf_val = 0
            # 保存每个高频词在每行文本的TF-IDF值
            feat_vec.append(tf_idf_val)

        # 赋值，
        # 将常用单词在每行文本的tf-idf值存到X对应的行里
        X[i, :] = np.array(feat_vec)
        # 获取每行label标签的情感值，并存储
        y[i] = int(r_data['label'])
    return X, y


def cal_acc(true_labels, pred_labels):
    """
        计算准确率
    """
    # 文档总行数：210000
    n_total = len(true_labels)

    # 判断模型预测结果和情感值，如果相等返回True，表示预测成功，否则返回False，表示预测失败。
    correct_list = [true_labels[i] == pred_labels[i] for i in range(n_total)]
    #[True, True, False, False, True....]

    # sum(correct_list) 统计所有成功的行，和总行数的商，表示预测准确率。
    acc = sum(correct_list) / n_total
    return acc
