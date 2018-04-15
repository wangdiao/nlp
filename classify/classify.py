import argparse
import sys

import tensorflow as tf

import classify
from classify.model import ClassifyModel
from classify.utils import build_word_index, create_vocab_tables, init_embedding, get_iterator, get_predict_iterator

FLAGS = None


def add_arguments(parser):
    parser.add_argument('--action', type=str, default="train")
    parser.add_argument('--out_dir', type=str, default='model/classify')


def run_main(argv):
    default_hparams = classify.create_hparams(FLAGS)
    hparams = classify.create_or_load_hparams(default_hparams)
    build_word_index(hparams.data, hparams.src, hparams.src_vocab_file, hparams.tgt_vocab_file)
    src_vocab_table, tgt_vocab_table = create_vocab_tables(hparams.src_vocab_file, hparams.tgt_vocab_file,
                                                           hparams.vocab_size, hparams.class_size)
    embedding = init_embedding(hparams.vocab_size, hparams.embeddings_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        # session = LocalCLIDebugWrapperSession(session)
        if FLAGS.action == 'train':
            input = get_iterator(hparams.src, hparams.buffer_size, hparams.random_seed,
                                 hparams.num_threads,
                                 hparams.max_len, src_vocab_table,
                                 tgt_vocab_table, hparams.vocab_size, hparams.batch_size,
                                 hparams.num_buckets)
            model = ClassifyModel(hparams, input, embedding, mode=tf.contrib.learn.ModeKeys.TRAIN)
            model.create_or_load(session, hparams.out_dir_model)
            model.train(session, hparams)
        elif FLAGS.action == 'infer':
            input, placeholder = get_predict_iterator(src_vocab_table)
            model = ClassifyModel(hparams, input, embedding, batch_size=1, mode=tf.contrib.learn.ModeKeys.INFER)
            model.create_or_load(session, hparams.out_dir_model)
            tag = model.infer(session, feed_dict={
                placeholder: ["明天 北京 的 天气 怎么样 ？"]})
            print(tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run_main, argv=[sys.argv[0]] + unparsed)
