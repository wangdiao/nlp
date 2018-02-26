import tensorflow as tf

import ner
from ner.model import NerModel
from ner.utils import get_iterator, create_vocab_tables, get_predict_iterator, init_embedding, build_word_index


def run_main(argv):
    hparams = ner.create_hparams()
    parser = ner.add_hparams(hparams)
    parser.parse_known_args(argv, namespace=hparams)
    action = hparams.action
    build_word_index(hparams.src, hparams.src_vocab_file, hparams.tgt, hparams.tgt_vocab_file)
    ner.init(hparams)
    ner.save_hparams(hparams)
    hparams.action = action
    src_vocab_table, tgt_vocab_table = create_vocab_tables(hparams.src_vocab_file, hparams.tgt_vocab_file,
                                                           hparams.vocab_size, 0)
    embedding = init_embedding(hparams.vocab_size, hparams.embeddings_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        if action == 'train':
            input = get_iterator(hparams.src, hparams.tgt, hparams.buffer_size, hparams.random_seed,
                                 hparams.num_threads,
                                 hparams.src_max_len, hparams.tgt_max_len, src_vocab_table,
                                 tgt_vocab_table, hparams.vocab_size, hparams.class_size, hparams.batch_size,
                                 hparams.num_buckets)
            model = NerModel(hparams, input, embedding)
            model.train(session, hparams)
        elif action == 'predict':
            input = get_predict_iterator(src_vocab_table, hparams.vocab_size, 1, None, hparams.pred)
            model = NerModel(hparams, input, embedding, batch_size=1)
            model.predict(session, hparams)


if __name__ == '__main__':
    tf.app.run(main=run_main)
