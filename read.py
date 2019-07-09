import os
import pandas as pd
import nltk
from tools import proc_text, split_train_test, get_word_list_from_data, extract_feat_from_data, cal_acc
from nltk.text import TextCollection
dataset_path = './dataset'
text_filenames = ['0_AI.txt', '1_AI.txt',
                  '2_AI.txt', '3_AI.txt']

# 原始数据的csv文件
output_text_filename = 'raw_AI_text.csv'

# 清洗好的文本数据文件
output_cln_text_filename = 'clean_AI_text.csv'
for text_filename in text_filenames:
    # 组合文件路径
    text_file = os.path.join(dataset_path, text_filename)

    # 获取标签，即0, 1, 2, 3
    label = int(text_filename[0])

    # 读取文本文件
    with open(text_file, 'r', encoding='utf-8') as f:
        # 将文本字符串按换行符(\n、\r、\r\n)分隔，返回包含每行数据的列表
        lines = f.read().splitlines()
        for l in lines:
            print(l)
