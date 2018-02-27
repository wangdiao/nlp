from ner import create_hparams, init, save_hparams
from ner.utils import build_word_index

if __name__ == '__main__':
    hparams = create_hparams()
    build_word_index(hparams.embedding_file, hparams.src_vocab_file, hparams.tgt, hparams.tgt_vocab_file)
    init(hparams)
    save_hparams(hparams)
