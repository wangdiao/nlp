import argparse
import codecs
import os

import tensorflow as tf

from ner.utils import get_file_line


def create_hparams():
    return tf.contrib.training.HParams(
        src='data/dependency_parsing/train.conll',
        validate='data/dependency_parsing/dev.conll',
        word_vocab_file='data/dependency_parsing/word_vocab.txt',
        c_vocab_file='data/dependency_parsing/c_vocab.txt',
        tgt_vocab_file='data/dependency_parsing/tgt_vocab.txt',
        max_len=None,
        num_units=256,
        num_layers=4,
        embeddings_size=128,
        buffer_size=200,
        random_seed=None,
        num_threads=2,
        dropout=0.2,
        forget_bias=0.8,
        batch_size=64,
        learning_rate=0.1,
        learning_rate_tau_steps=60000,
        learning_rate_tau_factor=2,
        start_decay_step=60000,
        decay_steps=10000,
        decay_factor=0.9,
        max_gradient_norm=5.0,
        num_train_steps=200000,
        num_buckets=3,
        out_dir='model/dependency_parsing',
        out_dir_model=None,
        out_dir_summary=None,
        word_vocab_size=0,
        c_vocab_size=0,
        tgt_vocab_size=0,
        action=''
    )


def add_hparams(hparams):
    parser = argparse.ArgumentParser(description='Train model.')
    for key, value in hparams.values().items():
        if value is not None:
            parser.add_argument('--' + key, type=type(value))
    return parser


def init(hparams):
    if not os.path.exists(hparams.out_dir):
        os.makedirs(hparams.out_dir)

    hparams.set_hparam('out_dir_model', hparams.out_dir)

    hparams_path = os.path.join(hparams.out_dir_model, 'config.json')
    if tf.gfile.Exists(hparams_path):
        with tf.gfile.GFile(hparams_path, "r") as f:
            hparams.parse_json(f.read())

    hparams.set_hparam('out_dir_summary', hparams.out_dir + '/summary')
    word_vocab_size = get_file_line(hparams.word_vocab_file)
    c_vocab_size = get_file_line(hparams.c_vocab_file)
    tgt_vocab_size = get_file_line(hparams.tgt_vocab_file)
    hparams.set_hparam('word_vocab_size', word_vocab_size)
    hparams.set_hparam('c_vocab_size', c_vocab_size)
    hparams.set_hparam('tgt_vocab_size', tgt_vocab_size)


def save_hparams(hparams):
    hparams_path = os.path.join(hparams.out_dir_model, 'config.json')
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_path, "wb")) as f:
        f.write(hparams.to_json())
