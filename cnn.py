import os
import pandas as pd
import nltk
from tools import proc_text, split_train_test, get_word_list_from_data, extract_feat_from_data, cal_acc
from nltk.text import TextCollection
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD, Adam
import keras
dataset_path = './dataset'
text_filenames = ['0_AI.txt', '1_AI.txt',
                  '2_AI.txt', '3_AI.txt']
output_text_filename = 'raw_AI_text.csv'
output_cln_text_filename = 'clean_AI_text.csv'
is_first_run = False
load_np = False


def read_and_save_to_csv():
    text_w_label_df_lst = []
    for text_filename in text_filenames:
        # 组合文件路径
        text_file = os.path.join(dataset_path, text_filename)
        label = int(text_filename[1])
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        labels = [label] * len(lines)
        text_series = pd.Series(lines)
        label_series = pd.Series(labels)
        text_w_label_df = pd.concat([label_series, text_series], axis=1)
        text_w_label_df_lst.append(text_w_label_df)
    result_df = pd.concat(text_w_label_df_lst, axis=0)
    result_df.columns = ['label', 'text']
    result_df.to_csv(os.path.join(dataset_path, output_text_filename),index=None, encoding='utf-8')


def run_main():
    if is_first_run:
        print('处理清洗文本数据中...', end=' ')
        read_and_save_to_csv()
        text_df = pd.read_csv(os.path.join(dataset_path, output_text_filename),
                              encoding='utf-8')
        text_df['text'] = text_df['text'].apply(proc_text)
        text_df = text_df[text_df['text'] != '']
        text_df.to_csv(os.path.join(dataset_path, output_cln_text_filename),
                       index=None, encoding='utf-8')
        print('完成，并保存结果。')
    print('加载处理好的文本数据')
    clean_text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename),
                                encoding='utf-8')
    train_text_df, test_text_df = split_train_test(clean_text_df)
    print('训练集中各类的数据个数：', train_text_df.groupby('label').size())
    print('测试集中各类的数据个数：', test_text_df.groupby('label').size())
    n_common_words = 1000
    all_words_in_train = get_word_list_from_data(train_text_df)
    print('统计词频...')
    print("总单词数",len(all_words_in_train))
    fdisk = nltk.FreqDist(all_words_in_train)
    print("词频",len(fdisk))
    common_words_freqs = fdisk.most_common(n_common_words)
    print('出现最多的{}个词是：'.format(n_common_words))
    for word, count in common_words_freqs:
        print('{}: {}次'.format(word, count))
    print()
    text_collection = TextCollection(train_text_df['text'].values.tolist())
    print('训练样本提取特征...', end=' ')
    if load_np:
        train_X = np.load("train_x.npy")
        print(train_X.shape)
        train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
        print(train_X.shape)
        train_y = np.load("train_y.npy")
        test_X = np.load("test_X.npy")
        test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
        test_y = np.load("test_y.npy")
    else:
        train_X, train_y = extract_feat_from_data(train_text_df, text_collection, common_words_freqs)
        np.save("train_x.npy", train_X)
        np.save("train_y.npy", train_y)
        print('完成')
        print()
        print('测试样本提取特征...', end=' ')
        test_X, test_y = extract_feat_from_data(test_text_df, text_collection, common_words_freqs)
        np.save("test_X.npy", test_X)
        np.save("test_y.npy", test_y)
        print('完成')
    print('训练模型...', end=' ')
    # 创建高斯朴素贝叶斯模型
    gnb = LogisticRegression(multi_class="ovr")
    model = get_model(n_common_words)
    onehot_train_y =  keras.utils.to_categorical(train_y, num_classes=4)
    onehot_test_y =  keras.utils.to_categorical(test_y, num_classes=4)
    # 向模型加载训练集特征数据，训练模型，
    gnb.fit(train_X, train_y)
    model.save_weights("model.h5")
    print('完成')
    print('测试模型...', end=' ')
    # 加载测试集特征数据，用来预测数据。
    test_pred = gnb.predict(test_X)
    print('完成')
    # 输出准确率
    print('准确率：', cal_acc(test_y, test_pred))


def cal_cnn_acc(true_labels, pred_labels):
    n_total = len(true_labels)
    correct_list = [np.argmax(true_labels[i]) == np.argmax(pred_labels[i]) for i in range(n_total)]
    acc = sum(correct_list) / n_total
    return acc


def get_model(n_common_words):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(n_common_words,)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                 metrics=['accuracy'])
    return model


if __name__ == '__main__':
    run_main()
