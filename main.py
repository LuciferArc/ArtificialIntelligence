# -*- coding: utf-8 -*-

import os
import pandas as pd
import nltk
from tools import proc_text, split_train_test, get_word_list_from_data, extract_feat_from_data, cal_acc
from nltk.text import TextCollection
from sklearn.naive_bayes import GaussianNB

dataset_path = './dataset'
text_filenames = ['0_AI.txt', '1_AI.txt',
                  '2_AI.txt', '3_AI.txt']

# 原始数据的csv文件
output_text_filename = 'raw_AI_text.csv'

# 清洗好的文本数据文件
output_cln_text_filename = 'clean_AI_text.csv'

# 处理和清洗文本数据的时间较长，通过设置is_first_run进行配置
# 如果是第一次运行需要对原始文本数据进行处理和清洗，需要设为True
# 如果之前已经处理了文本数据，并已经保存了清洗好的文本数据，设为False即可
is_first_run = False


def read_and_save_to_csv():
    """
        读取原始文本数据，将标签和文本数据保存成csv
    """

    # 存储所有向量化的DataFrame对象
    # 每个DataFrame对象表示一个文本数据
    text_w_label_df_lst = []
    #　循环获取每一个微博文本文件名
    for text_filename in text_filenames:
        # 组合文件路径
        text_file = os.path.join(dataset_path, text_filename)

        # 获取标签，即0, 1, 2, 3
        label = int(text_filename[0])

        # 读取文本文件
        with open(text_file, 'r', encoding='utf-8') as f:
            # 将文本字符串按换行符(\n、\r、\r\n)分隔，返回包含每行数据的列表
            lines = f.read().splitlines()

        #　生成一个向量，[0, 0, 0, 0 ....]
        labels = [label] * len(lines)

        # 当前文本内容的Series对象
        text_series = pd.Series(lines)
        # 当前文本的标签Series对象
        label_series = pd.Series(labels)

        # concat合并多个Series对象，返回一个DataFrame对象
        text_w_label_df = pd.concat([label_series, text_series], axis=1)
        # 将所有的数据集存到同一个列表里
        text_w_label_df_lst.append(text_w_label_df)

    result_df = pd.concat(text_w_label_df_lst, axis=0)

    # 保存成csv文件
    # 指定列名，第一个label，第二个text
    result_df.columns = ['label', 'text']
    # 将所有数据集写入到本地磁盘文件
    result_df.to_csv(os.path.join(dataset_path, output_text_filename),index=None, encoding='utf-8')


def run_main():
    """
        主函数
    """
    # 1. 数据读取，处理，清洗，准备
    if is_first_run:
        print('处理清洗文本数据中...', end=' ')
        # 如果是第一次运行需要对原始文本数据进行处理和清洗

        # 读取原始文本数据，将标签和文本数据保存成csv
        read_and_save_to_csv()

        # 读取处理好的csv文件，构造数据集
        text_df = pd.read_csv(os.path.join(dataset_path, output_text_filename),
                              encoding='utf-8')

        # 处理文本数据
        text_df['text'] = text_df['text'].apply(proc_text)

        # 过滤空字符串，去掉所有空行部分
        text_df = text_df[text_df['text'] != '']

        # 保存处理好的文本数据，文本预处理结束
        text_df.to_csv(os.path.join(dataset_path, output_cln_text_filename),
                       index=None, encoding='utf-8')
        print('完成，并保存结果。')



    # 2. 分割训练集、测试集
    print('加载处理好的文本数据')
    clean_text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename),
                                encoding='utf-8')
    # 分割训练集和测试集
    # 按每个情感值的80%做分割，
    train_text_df, test_text_df = split_train_test(clean_text_df)
    # 查看训练集测试集基本信息
    print('训练集中各类的数据个数：', train_text_df.groupby('label').size())
    print('测试集中各类的数据个数：', test_text_df.groupby('label').size())


    # 3. 特征提取
    # 计算词频
    n_common_words = 200

    # 将训练集中的单词拿出来统计词频
    print('统计词频...')

    # 获取训练集数据集里所有的词语的列表
    all_words_in_train = get_word_list_from_data(train_text_df)
    # 统计词频
    fdisk = nltk.FreqDist(all_words_in_train)

    # 获取词频排名前200个的词语的词频
    # 构建“常用单词列表”
    common_words_freqs = fdisk.most_common(n_common_words)
    print('出现最多的{}个词是：'.format(n_common_words))

    for word, count in common_words_freqs:
        print('{}: {}次'.format(word, count))
    print()

    # 在训练集上提取特征
    # 将text部分转换为list做为参数
    text_collection = TextCollection(train_text_df['text'].values.tolist())

    # 提取训练样本和测试样本的特征
    # _X 表示常用单词在每一行的tf-idf值，_y 表示情感值
    print('训练样本提取特征...', end=' ')
    train_X, train_y = extract_feat_from_data(train_text_df, text_collection, common_words_freqs)
    print('完成')
    print()

    print('测试样本提取特征...', end=' ')
    test_X, test_y = extract_feat_from_data(test_text_df, text_collection, common_words_freqs)
    print('完成')

    # 4. 训练模型Naive Bayes
    print('训练模型...', end=' ')
    # 创建高斯朴素贝叶斯模型
    gnb = GaussianNB()
    # 向模型加载训练集特征数据，训练模型，
    gnb.fit(train_X, train_y)
    print('完成')
    print()

    # 5. 预测
    print('测试模型...', end=' ')
    # 加载测试集特征数据，用来预测数据。
    test_pred = gnb.predict(test_X)
    # test_pred : ndarray : array([3., 3., 3., 2., 3., 3., 3., 0., 3., 3., 3., 2., 1. .....])

    print('完成')

    # 输出准确率
    print('准确率：', cal_acc(test_y, test_pred))
    # test_y : ndarray : array([3., 3., 3., 2., 3., 3., 3., 0., 3., 3., 3., 2., 1. .....])


if __name__ == '__main__':
    run_main()
