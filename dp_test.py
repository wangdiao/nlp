from dependency_parsing import create_hparams, init, save_hparams
from dependency_parsing.utils import build_word_index

if __name__ == '__main__':
    hparams = create_hparams()
    build_word_index(hparams.src, hparams.word_vocab_file,hparams.c_vocab_file, hparams.tgt_vocab_file)
    init(hparams)
    save_hparams(hparams)
