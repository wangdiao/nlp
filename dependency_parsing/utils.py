import collections
import os

import tensorflow as tf
from tensorflow.python.ops import lookup_ops


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "wi_ids", "wj_ids", "ci_ids", "cj_ids",
                            "deprel_ids",
                            "sequence_length"))):
    pass


def get_iterator(src_file, buffer_size, random_seed, num_threads, max_len, src_vocab_table,
                 c_vocab_table, tgt_vocab_table, src_vocab_size, c_vocab_size, tgt_vocab_size, batch_size, num_buckets,
                 once=False):
    wi_dataset = tf.data.TextLineDataset(src_file + '.wi')
    wj_dataset = tf.data.TextLineDataset(src_file + '.wj')
    ci_dataset = tf.data.TextLineDataset(src_file + '.ci')
    cj_dataset = tf.data.TextLineDataset(src_file + '.cj')
    deprel_dataset = tf.data.TextLineDataset(src_file + '.deprel')
    src_tgt_dataset = tf.data.Dataset.zip((wi_dataset, wj_dataset, ci_dataset, cj_dataset, deprel_dataset))

    if not once:
        src_tgt_dataset = src_tgt_dataset.repeat().shuffle(
            buffer_size, random_seed)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda wi, wj, ci, cj, deprel: (
            tf.string_split([wi]).values, tf.string_split([wj]).values, tf.string_split([ci]).values,
            tf.string_split([cj]).values, tf.string_split([deprel]).values),
        num_parallel_calls=num_threads)

    if max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda wi, wj, ci, cj, deprel: (wi[:max_len], wj[:max_len], ci[:max_len], cj[:max_len], deprel[:max_len]),
            num_parallel_calls=num_threads)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda wi, wj, ci, cj, deprel: (
            tf.cast(src_vocab_table.lookup(wi), tf.int32), tf.cast(src_vocab_table.lookup(wj), tf.int32),
            tf.cast(c_vocab_table.lookup(ci), tf.int32), tf.cast(c_vocab_table.lookup(cj), tf.int32),
            tf.cast(tgt_vocab_table.lookup(deprel), tf.int32)),
        num_parallel_calls=num_threads)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda wi, wj, ci, cj, deprel: (wi, wj, ci, cj, deprel, tf.size(wi)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    def batching_func(x):
        return x.padded_batch(batch_size,
                              # The entry is the source line rows;
                              # this has unknown-length vectors.  The last entry is
                              # the source row size; this is a scalar.
                              padded_shapes=(
                                  tf.TensorShape([None]),
                                  tf.TensorShape([None]),
                                  tf.TensorShape([None]),
                                  tf.TensorShape([None]),
                                  tf.TensorShape([None]),
                                  tf.TensorShape([])),
                              # Pad the source sequences with eos tokens.
                              # (Though notice we don't generally need to do this since
                              # later on we will be masking out calculations past the true sequence.
                              padding_values=(src_vocab_size + 1,  # src
                                              src_vocab_size + 1,
                                              c_vocab_size,
                                              c_vocab_size,
                                              tgt_vocab_size,
                                              0))  # src_len -- unused

    def key_func(unused_1, unused_2, unused_3, unused_4, unused_5, seq_len):
        # Calculate bucket_width by maximum source sequence length.
        # Pairs with length [0, bucket_width) go to bucket 0, length
        # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
        # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
        bucket_width = 8
        # Bucket sentence pairs by the length of their source sentence and target
        # sentence.
        bucket_id = tf.maximum(seq_len // bucket_width, seq_len // bucket_width)
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    batched_iter = batched_dataset.make_initializable_iterator()
    (wi_ids, wj_ids, ci_ids, cj_ids, deprel_ids, seq_len) = (
        batched_iter.get_next())

    return BatchedInput(
        initializer=batched_iter.initializer,
        wi_ids=wi_ids,
        wj_ids=wj_ids,
        ci_ids=ci_ids,
        cj_ids=cj_ids,
        deprel_ids=deprel_ids,
        sequence_length=seq_len)


def get_predict_iterator(src_vocab_table, c_vocab_table):
    wi_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name="wi_placeholder")
    wj_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name="wj_placeholder")
    ci_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name="ci_placeholder")
    cj_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name="cj_placeholder")
    wi_dataset = tf.data.Dataset.from_tensor_slices(wi_placeholder)
    wj_dataset = tf.data.Dataset.from_tensor_slices(wj_placeholder)
    ci_dataset = tf.data.Dataset.from_tensor_slices(ci_placeholder)
    cj_dataset = tf.data.Dataset.from_tensor_slices(cj_placeholder)
    src_tgt_dataset = tf.data.Dataset.zip((wi_dataset, wj_dataset, ci_dataset, cj_dataset))
    src_tgt_dataset = src_tgt_dataset.map(
        lambda wi, wj, ci, cj: (
            tf.string_split([wi]).values, tf.string_split([wj]).values, tf.string_split([ci]).values,
            tf.string_split([cj]).values))
    src_tgt_dataset = src_tgt_dataset.map(
        lambda wi, wj, ci, cj: (
            tf.cast(src_vocab_table.lookup(wi), tf.int32), tf.cast(src_vocab_table.lookup(wj), tf.int32),
            tf.cast(c_vocab_table.lookup(ci), tf.int32), tf.cast(c_vocab_table.lookup(cj), tf.int32)))
    src_tgt_dataset = src_tgt_dataset.map(
        lambda wi, wj, ci, cj: (wi, wj, ci, cj, tf.size(wi)))
    batched_dataset = src_tgt_dataset.batch(1)
    batched_iter = batched_dataset.make_initializable_iterator()
    (wi_ids, wj_ids, ci_ids, cj_ids, seq_len) = (
        batched_iter.get_next())

    return BatchedInput(
        initializer=batched_iter.initializer,
        wi_ids=wi_ids,
        wj_ids=wj_ids,
        ci_ids=ci_ids,
        cj_ids=cj_ids,
        deprel_ids=None,
        sequence_length=seq_len), wi_placeholder, wj_placeholder, ci_placeholder, cj_placeholder


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


def build_word_index(src_file, word_vocab_file, c_vocab_file, tgt_vocab_file):
    '''
        生成单词列表，并存入文件之中。
    :return:
    '''
    if not os.path.exists(src_file):
        print('source file does not exist, please check your file path ')
        return

    print('building word index...')
    if not os.path.exists(word_vocab_file):
        with open(src_file, 'r', encoding="UTF-8") as f:
            dict_word = {}
            dict_c = {}
            dict_deprel = {}
            for line in f:
                line = line.strip()
                if line != '':
                    word_arr = line.split('\t')
                    word_struct = {"index": int(word_arr[0]), "word": word_arr[1], "cpostag": word_arr[3],
                                   "head": int(word_arr[6]), "deprel": word_arr[7]}
                    word = word_struct["word"]
                    dict_word[word] = dict_word.get(word, 0) + 1
                    c = word_struct["cpostag"]
                    dict_c[c] = dict_c.get(c, 0) + 1
                    deprel = word_struct["deprel"]
                    dict_deprel[deprel] = dict_deprel.get(deprel, 0) + 1
            top_words = sorted(dict_word.items(), key=lambda s: s[1], reverse=True)
            with open(word_vocab_file, 'w', encoding="UTF-8") as s_vocab:
                s_vocab.write('ROOT\n')
                for word, frequence in top_words:
                    s_vocab.write(word + '\n')

            top_c = sorted(dict_c.items(), key=lambda s: s[1], reverse=True)
            with open(c_vocab_file, 'w', encoding="UTF-8") as c_vocab:
                c_vocab.write('ROOT\n')
                for c, frequence in top_c:
                    c_vocab.write(c + '\n')

            top_deprel = sorted(dict_deprel.items(), key=lambda s: s[1], reverse=True)
            with open(tgt_vocab_file, 'w', encoding="UTF-8") as t_vocab:
                t_vocab.write('NONE\n')
                for deprel, frequence in top_deprel:
                    t_vocab.write(deprel + '+\n')
                    t_vocab.write(deprel + '-\n')
    else:
        print('source vocabulary file has already existed, continue to next stage.')
    generate_wcd(src_file)


def generate_wcd(src_file):
    with open(src_file, 'r', encoding="UTF-8") as f, open(src_file + ".wi", 'w', encoding="UTF-8") as wif, open(
            src_file + ".wj", 'w', encoding="UTF-8") as wjf, open(src_file + ".ci", 'w', encoding="UTF-8") as cif, open(
            src_file + ".cj", 'w', encoding="UTF-8") as cjf, open(src_file + ".deprel", 'w',
                                                                  encoding="UTF-8") as deprelf:
        seq = []
        for line in f:
            line = line.strip()
            if line != '':
                word_arr = line.split('\t')
                word_struct = {"index": int(word_arr[0]), "word": word_arr[1], "cpostag": word_arr[3],
                               "head": int(word_arr[6]), "deprel": word_arr[7]}
                seq.append(word_struct)
            else:
                if len(seq) == 0:
                    continue
                word_seq = ['ROOT'] + [ws['word'] for ws in seq]
                cpostag_seq = ['ROOT'] + [ws['cpostag'] for ws in seq]
                res_word_i = []
                res_word_j = []
                res_cpostag_i = []
                res_cpostag_j = []
                res_deprel = []
                for i in range(len(word_seq)):
                    for j in range(len(word_seq)):
                        if i < j:
                            res_word_i.append(word_seq[i])
                            res_word_j.append(word_seq[j])
                            res_cpostag_i.append(cpostag_seq[i])
                            res_cpostag_j.append(cpostag_seq[j])
                            deprel = "NONE"
                            for ws in seq:
                                if ws["index"] == i and ws["head"] == j:
                                    deprel = ws["deprel"] + "+"
                                    break
                                elif ws["index"] == j and ws["head"] == i:
                                    deprel = ws["deprel"] + "-"
                                    break
                            res_deprel.append(deprel)
                wif.write(' '.join(res_word_i) + '\n')
                wjf.write(' '.join(res_word_j) + '\n')
                cif.write(' '.join(res_cpostag_i) + '\n')
                cjf.write(' '.join(res_cpostag_j) + '\n')
                deprelf.write(' '.join(res_deprel) + '\n')
                seq = []


def generate_conll(src_file):
    pass


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
