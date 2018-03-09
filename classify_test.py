from classify.utils import build_word_index

if __name__ == '__main__':
    build_word_index('data/classify/data.txt', 'data/classify/src.txt', 'data/classify/word_vocab.txt',
                     'data/classify/label_vocab.txt')
