import collections
import os
import re

import jieba
import tensorflow as tf
from tensorflow.python.ops import lookup_ops


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target",
                            "source_sequence_length"))):
    pass


def get_iterator(src_file, buffer_size, random_seed, num_threads, max_len, src_vocab_table,
                 tgt_vocab_table, vocab_size, batch_size, num_buckets):
    src_dataset = tf.data.TextLineDataset(src_file)

    def split_src_tgt(src):
        data = tf.string_split([src], '|').values
        return data[0], data[-1]

    src_tgt_dataset = src_dataset.map(
        split_src_tgt,
        num_parallel_calls=num_threads)
    src_tgt_dataset = src_tgt_dataset.repeat().shuffle(
        buffer_size, random_seed)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.string_split([src]).values, tgt),
        num_parallel_calls=num_threads)

    if max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:max_len], tgt),
            num_parallel_calls=num_threads)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_threads)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            src, tgt, tf.size(src)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    def batching_func(x):
        return x.padded_batch(batch_size,
                              # The entry is the source line rows;
                              # this has unknown-length vectors.  The last entry is
                              # the source row size; this is a scalar.
                              padded_shapes=(
                                  tf.TensorShape([None]),  # src
                                  tf.TensorShape([]),  # tgt
                                  tf.TensorShape([])),  # src_len
                              # Pad the source sequences with eos tokens.
                              # (Though notice we don't generally need to do this since
                              # later on we will be masking out calculations past the true sequence.
                              padding_values=(vocab_size + 1,  # src
                                              0,
                                              0))  # src_len -- unused

    def key_func(unused_1, unused_2, src_len):
        # Calculate bucket_width by maximum source sequence length.
        # Pairs with length [0, bucket_width) go to bucket 0, length
        # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
        # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
        bucket_width = 8
        # Bucket sentence pairs by the length of their source sentence and target
        # sentence.
        bucket_id = src_len // bucket_width
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    batched_iter = batched_dataset.make_initializable_iterator()
    src_ids, tgt_ids, src_seq_len = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target=tgt_ids,
        source_sequence_length=src_seq_len)


def get_predict_iterator(src_vocab_table):
    sentence_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name="sentence_placeholder")
    src_tgt_dataset = tf.data.Dataset.from_tensor_slices(sentence_placeholder)
    src_tgt_dataset = src_tgt_dataset.map(
        lambda word: tf.string_split([word]).values)
    src_tgt_dataset = src_tgt_dataset.map(
        lambda word: tf.cast(src_vocab_table.lookup(word), tf.int32))
    src_tgt_dataset = src_tgt_dataset.map(lambda word_id: (word_id, tf.size(word_id)))
    batched_dataset = src_tgt_dataset.batch(1)
    batched_iter = batched_dataset.make_initializable_iterator()
    word_id, seq_len = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=word_id,
        target=None,
        source_sequence_length=seq_len), sentence_placeholder


def get_online_iterator(buffer_size, num_threads, max_len, src_vocab_table,
                        tgt_vocab_table, batch_size):
    sentence_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name="sentence_placeholder")
    label_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name="label_placeholder")
    src_dataset = tf.data.Dataset.from_tensor_slices(sentence_placeholder)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_placeholder)

    src_tgt_dataset = src_dataset.zip((src_dataset, label_dataset))
    src_tgt_dataset = src_tgt_dataset.repeat()
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.string_split([src]).values, tgt),
        num_parallel_calls=num_threads)

    if max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:max_len], tgt),
            num_parallel_calls=num_threads)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_threads)

    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src, tgt, tf.size(src)), num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)
    batched_dataset = src_tgt_dataset.batch(batch_size)
    batched_iter = batched_dataset.make_initializable_iterator()
    src_ids, tgt_ids, src_seq_len = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target=tgt_ids,
        source_sequence_length=src_seq_len), sentence_placeholder, label_placeholder


def build_word_index(data_file, src_file, src_vocab_file, tgt_vocab_file):
    '''
        生成单词列表，并存入文件之中。
    :return:
    '''
    if not os.path.exists(src_file):
        with open(data_file, 'r', encoding="UTF-8") as f, open(src_file, 'w', encoding="UTF-8") as srcf:
            for line in f:
                line = line.strip()
                if line == '':
                    break
                index = line.find('|')
                if index < 0:
                    break
                sentence = line[0:index].strip()
                sentence = re.sub(r'\[(.*?)\]\(.*?\)', lambda matched: matched.group(1), sentence)
                label = line[index + 1:].strip()
                words = jieba.cut(sentence)
                srcf.write(' '.join(list(words)) + '|' + label + '\n')

    print('building word index...')
    if not os.path.exists(src_vocab_file):
        with open(src_file, 'r', encoding="UTF-8") as f:
            dict_word = {}
            dict_label = {}
            for line in f:
                line = line.strip()
                if line == '':
                    break
                index = line.find('|')
                if index < 0:
                    break
                sentence = line[0:index].strip()
                label = line[index + 1:].strip()
                word_arr = sentence.split(' ')
                for w in word_arr:
                    dict_word[w] = dict_word.get(w, 0) + 1
                dict_label[label] = dict_label.get(label, 0) + 1

            top_words = sorted(dict_word.items(), key=lambda s: s[1], reverse=True)
            with open(src_vocab_file, 'w', encoding="UTF-8") as s_vocab:
                for word, frequence in top_words:
                    s_vocab.write(word + '\n')

            top_labels = sorted(dict_label.items(), key=lambda s: s[1], reverse=True)
            with open(tgt_vocab_file, 'w', encoding="UTF-8") as t_vocab:
                for label, frequence in top_labels:
                    t_vocab.write(label + '\n')
    else:
        print('source vocabulary file has already existed, continue to next stage.')


def get_file_line(src_vocab_file):
    size = 0
    with open(src_vocab_file, 'r', encoding="UTF-8") as f:
        for content in f:
            content = content.strip()
            if content != '':
                size += 1
    return size


def file_content_iterator(file_name):
    with open(file_name, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            yield line.strip()


def create_vocab_tables(src_vocab_file, tgt_vocab_file, src_unknown_id, tgt_unknown_id, share_vocab=False):
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=src_unknown_id)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(
            tgt_vocab_file, default_value=tgt_unknown_id)
    return src_vocab_table, tgt_vocab_table


def init_embedding(vocab_size, embeddings_size):
    embeddings = tf.Variable(
        tf.random_uniform([vocab_size + 2, embeddings_size], -1, 1), name="embeddings")
    return embeddings


def equal_float(x, y, dtype=tf.float32):
    return tf.cast(tf.equal(x, y), dtype)


def not_equal_float(x, y, dtype=tf.float32):
    return tf.subtract(1, tf.cast(tf.equal(x, y), dtype))
