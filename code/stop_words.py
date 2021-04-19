from nltk.corpus import stopwords
from stemming import get_tokens


def remove_stopwords(texts):
    stopwords_arr = stopwords.words('english')
    sentences = []
    for text in texts:
        tokens = get_tokens(text)
        without_stopwords = [x for x in tokens if x not in stopwords_arr]
        sentences.append(' '.join(without_stopwords))
    return sentences

if __name__ == "__main__":
    texts = ["My car is really fast",
             "I like driving cars",
             "This flower are beautiful",
             "Taking care of flowers is realxing",
             "Cars are my hobby",
             "Flowers are my hobby"]

    print(remove_stopwords(texts))
