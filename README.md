# ArtificialIntelligence

cnn.py为使用cnn完成的自然语言情感分析，main.py则是使用高斯朴素贝叶斯模型进行分析，两种代码大致程序不一样，但运行结果是一样的。read.py文件为读取数据集中内容的操作，tools.py为两种程序所需的一部分功能函数代码。

这里我们使用的是从网络上搜集到的一些微博上的评论作为数据集。0_AI.txt、1_AI.txt、2_AI.txt、3_AI.txt为数据集，clean_AI_text.csv为清洗过后的数据集内容，raw_AI_text.csv为原数据集内容，在运行前需将这六个文件全部放在工程的dataset文件夹下。
test_x,text_y,train_x,train_y四个文件放在百度网盘https://pan.baidu.com/s/1KgJaCwkmh9REWWocD49MwQ，提取码为ih2b。

为了保证停用词的更加完整，我们从网上搜集了六种停用词表（库）对停用词进行完善。六个停用词文件直接放在工程文件夹下即可。

0：喜悦
1：愤怒
2：厌恶
3：低落
对应不同类别的感情
