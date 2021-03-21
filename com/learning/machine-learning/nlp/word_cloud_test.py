import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud, ImageColorGenerator
import cv2

if __name__ == "__main__":
    df = pd.read_csv('./data/society_news.csv', index_col=0, encoding='utf-8').astype('str')
    df['content'] = df['content'].apply(lambda x: jieba.lcut(x))

    stopwords = pd.read_csv('./data/stopwords.txt', quoting=3, sep='\t', names=['stopwords'], encoding='utf-8')
    words = []
    for content in df['content'].values:
        for word in content:
            if word not in stopwords['stopwords'].values and len(word) > 1:
                words.append(word)

    words_df = pd.DataFrame(words, columns=['words'])
    words_df['count'] = np.arange(len(words_df))
    words_group = words_df.groupby('words').count()
    words_group.sort_values(by='count', ascending=False, inplace=True)
    wordcloud = WordCloud(font_path='./data/simhei.ttf', max_words=50)
    word_freq = {i: j for i, j in zip(words_group.index.values, words_group['count'].values)}

    wordcloud = wordcloud.fit_words(word_freq)
    plt.imshow(wordcloud)


    image = cv2.imread('./data/python.jpg')
    wordcloud = WordCloud(font_path = './data/simhei.ttf',background_color='white',max_words = 50,mask = image)
    wordcloud.fit_words(word_freq)
    image_color = ImageColorGenerator(image)
    plt.imshow(wordcloud.recolor(color_func=image_color))
    plt.show()
    print("end")