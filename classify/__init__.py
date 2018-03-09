import codecs
import json
import os

import tensorflow as tf

from classify import classify
from classify.model import ClassifyModel
from classify.utils import get_file_line, create_vocab_tables, init_embedding, get_predict_iterator, get_online_iterator


def create_hparams(flags):
    return tf.contrib.training.HParams(
        data='data/classify/data.txt',
        src='data/classify/src.txt',
        src_vocab_file='data/classify/word_vocab.txt',
        tgt_vocab_file='data/classify/label_vocab.txt',
        max_len=None,
        num_units=128,
        num_layers=4,
        embeddings_size=32,
        buffer_size=200,
        random_seed=None,
        num_threads=2,
        dropout=0.2,
        forget_bias=0.8,
        batch_size=16,
        learning_rate=0.1,
        max_gradient_norm=5.0,
        num_train_steps=200000,
        num_buckets=3,
        out_dir=flags.out_dir,
    )


def create_or_load_hparams(default_hparams, save_hparams=True):
    hparams = load_hparams(default_hparams.out_dir)
    if hparams is None:
        hparams = default_hparams
        if not os.path.exists(hparams.out_dir):
            os.makedirs(hparams.out_dir)
        hparams.add_hparam('out_dir_model', hparams.out_dir)
        hparams.add_hparam('out_dir_summary', hparams.out_dir + '/summary')
        vocab_size = get_file_line(hparams.src_vocab_file)
        class_size = get_file_line(hparams.tgt_vocab_file)
        hparams.add_hparam('vocab_size', vocab_size)
        hparams.add_hparam('class_size', class_size)

    # Save HParams
    if save_hparams:
        hparams_path = os.path.join(hparams.out_dir_model, 'config.json')
        with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_path, "wb")) as f:
            f.write(hparams.to_json())

    return hparams


def load_hparams(out_dir):
    # Load hparams from an existing model directory.
    hparams_path = os.path.join(out_dir, 'config.json')
    if os.path.exists(hparams_path):
        with open(hparams_path, "r", encoding="UTF-8") as f:
            hparams_values = json.load(f)
            hparams = tf.contrib.training.HParams(**hparams_values)
            return hparams
    return None


class Model(object):
    def __init__(self, model_dir) -> None:
        self.hparams = load_hparams(model_dir)
        self.src_vocab_table, self.tgt_vocab_table = create_vocab_tables(self.hparams.src_vocab_file,
                                                                         self.hparams.tgt_vocab_file,
                                                                         self.hparams.vocab_size,
                                                                         self.hparams.class_size)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        input, placeholder = get_predict_iterator(self.src_vocab_table)
        self.placeholder = placeholder
        self.embedding = init_embedding(self.hparams.vocab_size, self.hparams.embeddings_size)
        self.reload_model()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.session.close()

    def predict(self, sentense):
        return self.model.predict(self.session, feed_dict={
            self.placeholder: [sentense]})

    def train_online(self, sentenses, labels, batch_size, num_train_steps):
        embedding = init_embedding(self.hparams.vocab_size, self.hparams.embeddings_size)
        input, sentence_placeholder, label_placeholder = get_online_iterator(self.hparams.buffer_size,
                                                                             self.hparams.num_threads,
                                                                             self.hparams.max_len, self.src_vocab_table,
                                                                             self.tgt_vocab_table, batch_size)
        model = ClassifyModel(self.hparams, input, embedding, batch_size=batch_size)
        model.create_or_load(self.session, self.hparams.out_dir_model)
        model.train_online(self.session, self.hparams, num_train_steps, feed_dict={
            sentence_placeholder: sentenses, label_placeholder: labels
        })
        self.reload_model()

    def reload_model(self):
        self.model = ClassifyModel(self.hparams, input, self.embedding, batch_size=1)
        self.model.create_or_load(self.session, self.hparams.out_dir_model)
