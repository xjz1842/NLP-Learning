
import jieba
import jieba.analyse as analyse
import jieba.posseg as pseg


if __name__ == "__main__":
    seq_list = jieba.cut("我在学习自然语言处理", cut_all=True)
    print(seq_list)
    print("/".join(seq_list))

    seq_list = jieba.cut("我在学习自然语言处理", cut_all=False)
    print(seq_list)
    print("/".join(seq_list))

    seq_list = jieba.cut_for_search("我在学习自然语言处理")
    print(seq_list)
    print("/".join(seq_list))

    result_lcut = jieba.lcut("小明硕士毕业于中国科学院计算所，后在哈佛大学深造")
    print(result_lcut)
    print(" ".join(result_lcut))

    lines = open('NBA.txt').read()
    print("  ".join(analyse.extract_tags(lines, topK=20, withWeight=False, allowPOS=())))

    print("  ".join(analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))
    print("---------------------我是分割线----------------")
    print("  ".join(analyse.textrank(lines, topK=20, withWeight=False, allowPOS=('ns', 'n'))))

    words = pseg.cut("我爱自然语言处理")
    for word, flag in words:
        print('%s %s' % (word, flag))

    print("这是默认模式的tokenize")
    result = jieba.tokenize(u'自然语言处理非常有用')
    for tk in result:
        print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

    print("\n-----------我是神奇的分割线------------\n")
    print("这是搜索模式的tokenize")
    result = jieba.tokenize(u'自然语言处理非常有用', mode='search')
    for tk in result:
        print("%s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))




