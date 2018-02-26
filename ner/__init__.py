import argparse
import codecs
import os

import tensorflow as tf

from ner.model import NerModel
from ner.utils import get_file_line, get_iterator, create_vocab_tables, load_word2vec_embedding, get_predict_iterator


def create_hparams():
    return tf.contrib.training.HParams(
        src='data/ner/source.txt',
        tgt='data/ner/target.txt',
        pred='data/ner/predict.txt',
        src_vocab_file='data/ner/src_vocab.txt',
        tgt_vocab_file='data/ner/tgt_vocab.txt',
        embedding_file='data/wiki.zh.vec',
        src_max_len=None,
        tgt_max_len=None,
        num_units=512,
        num_layers=4,
        embeddings_size=300,
        buffer_size=200,
        random_seed=None,
        num_threads=2,
        dropout=0.8,
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
        out_dir='model/ner',
        out_dir_model=None,
        out_dir_summary=None,
        vocab_size=0,
        class_size=0,
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
    vocab_size = get_file_line(hparams.src_vocab_file)
    class_size = get_file_line(hparams.tgt_vocab_file)
    hparams.set_hparam('vocab_size', vocab_size)
    hparams.set_hparam('class_size', class_size)


def save_hparams(hparams):
    hparams_path = os.path.join(hparams.out_dir_model, 'config.json')
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_path, "wb")) as f:
        f.write(hparams.to_json())
