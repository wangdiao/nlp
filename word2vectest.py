import gensim
from gensim.models.keyedvectors import Word2VecKeyedVectors

if __name__ == '__main__':
    word2vec = Word2VecKeyedVectors.load_word2vec_format(fname="../data/wv2/zhwiki")
    print(word2vec.wv[u'北京'])
    print(word2vec.wv.most_similar(u'北京'))