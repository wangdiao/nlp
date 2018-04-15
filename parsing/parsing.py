from dependency_parsing import *
from dependency_parsing.model import DependencyParsingModel
from dependency_parsing.utils import *


def run_main(argv):
    hparams = create_hparams()
    parser = add_hparams(hparams)
    parser.parse_known_args(argv, namespace=hparams)
    action = hparams.action
    build_word_index(hparams.src, hparams.word_vocab_file, hparams.c_vocab_file, hparams.tgt_vocab_file)

    init(hparams)
    save_hparams(hparams)
    hparams.action = action
    word_vocab_table = lookup_ops.index_table_from_file(hparams.word_vocab_file, default_value=hparams.word_vocab_size)
    c_vocab_table = lookup_ops.index_table_from_file(hparams.c_vocab_file)
    tgt_vocab_table = lookup_ops.index_table_from_file(hparams.tgt_vocab_file)
    embedding = init_embedding(hparams.word_vocab_size, hparams.embeddings_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        if action == 'train':
            input = get_iterator(hparams.src, hparams.buffer_size, hparams.random_seed,
                                 hparams.num_threads,
                                 hparams.max_len, word_vocab_table, c_vocab_table,
                                 tgt_vocab_table, hparams.word_vocab_size, hparams.c_vocab_size, hparams.tgt_vocab_size,
                                 hparams.batch_size,
                                 hparams.num_buckets)
            model = DependencyParsingModel(hparams, input, embedding)
            model.train(session, hparams)
        elif action == 'validate':
            generate_wcd(hparams.validate)
            input = get_iterator(hparams.validate, hparams.buffer_size, hparams.random_seed,
                                 hparams.num_threads,
                                 hparams.max_len, word_vocab_table, c_vocab_table,
                                 tgt_vocab_table, hparams.word_vocab_size, hparams.c_vocab_size, hparams.tgt_vocab_size,
                                 hparams.batch_size,
                                 hparams.num_buckets, once=True)
            model = DependencyParsingModel(hparams, input, embedding)
            model.validate(session, hparams)
        elif action == 'predict':
            input, wi_placeholder, wj_placeholder, ci_placeholder, cj_placeholder = get_predict_iterator(
                word_vocab_table, c_vocab_table)
            model = DependencyParsingModel(hparams, input, embedding, batch_size=1)
            words = ['ROOT'] + ['世界', '第', '八', '大', '奇迹', '出现']
            cs = ['ROOT'] + ['n', 'm', 'm', 'a', 'n', 'v']
            res_word_i = []
            res_word_j = []
            res_cpostag_i = []
            res_cpostag_j = []
            res_i = []
            res_j = []
            words_len = len(words)
            for i in range(words_len):
                for j in range(words_len):
                    if i < j:
                        res_word_i.append(words[i])
                        res_word_j.append(words[j])
                        res_cpostag_i.append(cs[i])
                        res_cpostag_j.append(cs[j])
                        res_i.append(i)
                        res_j.append(j)
            tags = model.predict(session, hparams, feed_dict={
                wi_placeholder: [' '.join(res_word_i)],
                wj_placeholder: [' '.join(res_word_j)],
                ci_placeholder: [' '.join(res_cpostag_i)],
                cj_placeholder: [' '.join(res_cpostag_j)]
            })
            for word_index, word in enumerate(words):
                head = -1
                deprel = '_'
                for index, tag in enumerate(tags):
                    if res_i[index] == word_index and tag.endswith('+'):
                        head = res_j[index]
                        deprel = tag[:-1]
                        break
                    elif res_j[index] == word_index and tag.endswith('-'):
                        head = res_i[index]
                        deprel = tag[:-1]
                        break
                print('%d\t%s\t%s\t%s\t%s\t_\t%d\t%s' % (
                    word_index, word, word, cs[word_index], cs[word_index], head, deprel))


if __name__ == '__main__':
    tf.app.run(main=run_main)
