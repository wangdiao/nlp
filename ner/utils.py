import collections
import os
from itertools import islice

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "source_sequence_length",
                            "target_sequence_length"))):
    pass


def get_iterator(src_file, tgt_file, buffer_size, random_seed, num_threads, src_max_len, tgt_max_len, src_vocab_table,
                 tgt_vocab_table, vocab_size, class_size, batch_size, num_buckets):
    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.repeat().shuffle(
        buffer_size, random_seed)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_threads)

    # src_tgt_dataset = src_tgt_dataset.filter(
    #     lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_threads)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_threads)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_threads)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in: (
            src, tgt_in, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    def batching_func(x):
        return x.padded_batch(batch_size,
                              # The entry is the source line rows;
                              # this has unknown-length vectors.  The last entry is
                              # the source row size; this is a scalar.
                              padded_shapes=(
                                  tf.TensorShape([None]),  # src
                                  tf.TensorShape([None]),  # tgt
                                  tf.TensorShape([]),  # src_len
                                  tf.TensorShape([])),  # tgt_len
                              # Pad the source sequences with eos tokens.
                              # (Though notice we don't generally need to do this since
                              # later on we will be masking out calculations past the true sequence.
                              padding_values=(vocab_size + 1,  # src
                                              class_size,
                                              0,
                                              0))  # src_len -- unused

    def key_func(unused_1, unused_2, src_len, tgt_len):
        # Calculate bucket_width by maximum source sequence length.
        # Pairs with length [0, bucket_width) go to bucket 0, length
        # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
        # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
        bucket_width = 8
        # Bucket sentence pairs by the length of their source sentence and target
        # sentence.
        bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, src_seq_len, tgt_seq_len) = (
        batched_iter.get_next())

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)


def get_predict_iterator(src_vocab_table, vocab_size, batch_size, max_len, pred_file):
    pred_dataset = tf.contrib.data.TextLineDataset(pred_file)
    pred_dataset = pred_dataset.map(
        lambda src: tf.string_split([src]).values)
    if max_len:
        pred_dataset = pred_dataset.map(lambda src: src[:max_len])

    pred_dataset = pred_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

    pred_dataset = pred_dataset.map(lambda src: (src, tf.size(src)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([])),  # src_len
            padding_values=(vocab_size + 1,  # src
                            0))  # src_len -- unused

    batched_dataset = batching_func(pred_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len) = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=None
    )


def write_to_vocab(file, vocab_file):
    with open(file, 'r', encoding="UTF-8") as f:
        dict_word = {}
        for line in f:
            line = line.strip()
            if line != '':
                word_arr = line.split()
                for w in word_arr:
                    dict_word[w] = dict_word.get(w, 0) + 1

        top_words = sorted(dict_word.items(), key=lambda s: s[1], reverse=True)
        with open(vocab_file, 'w', encoding="UTF-8") as s_vocab:
            for word, frequence in top_words:
                s_vocab.write(word + '\n')


def build_word_index(src_file, src_vocab_file, tgt_file, tgt_vocab_file):
    '''
        生成单词列表，并存入文件之中。
    :return:
    '''
    if not os.path.exists(src_file):
        print('source file does not exist, please check your file path ')
        return

    print('building word index...')
    if not os.path.exists(src_vocab_file):
        write_to_vocab(src_file, src_vocab_file)
    else:
        print('source vocabulary file has already existed, continue to next stage.')

    if not os.path.exists(tgt_vocab_file):
        write_to_vocab(tgt_file, tgt_vocab_file)
    else:
        print('target vocabulary file has already existed, continue to next stage.')


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


def load_word2vec_embedding(vocab_size, embeddings_size, word_embedding_file):
    '''
        加载外接的词向量。
        :return:
    '''
    print('loading word embedding, it will take few minutes...')
    embeddings = init_embedding(vocab_size, embeddings_size)
    # 保证每次随机出来的数一样。
    rng = np.random.RandomState(23455)
    unknown = np.asarray(rng.normal(size=embeddings_size))
    padding = np.asarray(rng.normal(size=embeddings_size))
    with open(word_embedding_file, 'r', encoding="UTF-8") as f:
        for index, line in enumerate(islice(f, 1, None)):
            values = line.split()
            try:
                coefs = np.asarray(values[1:], dtype='float32')  # 取向量
                embeddings[index] = coefs  # 将词和对应的向量存到字典里
            except ValueError:
                print(index, values[0], values[1:])
    # 顺序不能错，这个和unkown_id和padding id需要一一对应。
    embeddings[-2] = unknown
    embeddings[-1] = padding

    return tf.get_variable("embeddings", dtype=tf.float32,
                           shape=[vocab_size + 2, embeddings_size],
                           initializer=tf.constant_initializer(embeddings), trainable=False)


def equal_float(x, y, dtype=tf.float32):
    return tf.cast(tf.equal(x, y), dtype)


def not_equal_float(x, y, dtype=tf.float32):
    return tf.subtract(1, tf.cast(tf.equal(x, y), dtype))


def f1_score(y, y_, epsilon=1e-6, positive=None, negative=None):
    '''
    fn：False Negative,被判定为负样本，但事实上是正样本。
    fp：False Positive,被判定为正样本，但事实上是负样本。
    tn：True Negative,被判定为负样本，事实上也是负样本。
    tp：True Positive,被判定为正样本，事实上也是正样本。
    precesion：查准率，正确的正样本个数占被判断为正样本结果的比例。 precision = tp / (tp + fp)
    recall：查全率，正确的正样本个数占所有正样本的比例。 recall = tp / (tp + fn)
    f1_score = (2 * (precision * recall)) / (precision + recall)
    :return:
    '''
    if positive is None and negative is None:
        negative = 0
    if positive is not None:
        fn = equal_float(not_equal_float(y_, positive, dtype=tf.int32),
                         equal_float(y, positive, dtype=tf.int32)) + epsilon
        fp = equal_float(equal_float(y_, positive, dtype=tf.int32),
                         not_equal_float(y, positive, dtype=tf.int32)) + epsilon
        tn = equal_float(not_equal_float(y_, positive, dtype=tf.int32),
                         not_equal_float(y, positive, dtype=tf.int32)) + epsilon
        tp = equal_float(equal_float(y_, positive, dtype=tf.int32), equal_float(y, positive, dtype=tf.int32)) + epsilon
    else:
        fn = equal_float(equal_float(y_, negative, dtype=tf.int32),
                         not_equal_float(y, negative, dtype=tf.int32)) + epsilon
        fp = equal_float(not_equal_float(y_, negative, dtype=tf.int32),
                         equal_float(y, negative, dtype=tf.int32)) + epsilon
        tn = equal_float(equal_float(y_, negative, dtype=tf.int32), equal_float(y, negative, dtype=tf.int32)) + epsilon
        tp = equal_float(not_equal_float(y_, negative, dtype=tf.int32),
                         not_equal_float(y, negative, dtype=tf.int32)) + epsilon

    precision = tf.divide(tp, tf.add(tp, fp))
    recall = tf.divide(tp, tf.add(tp, fn))
    # return (2 * (precision * recall)) / (precision + recall)
    return tf.divide(tf.multiply(2, tf.multiply(precision, recall)), tf.add(precision, recall))
