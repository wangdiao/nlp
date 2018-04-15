import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from ner.utils import equal_float


class DependencyParsingModel(object):
    def __init__(self, hparams, input, embedding, batch_size=None) -> None:
        self.input = input
        self.embedding = embedding
        self.embeddings_size = hparams.embeddings_size
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        if batch_size is None:
            self.batch_size = hparams.batch_size
        else:
            self.batch_size = batch_size
        self._build_graph(hparams)
        self.saver = tf.train.Saver(tf.global_variables())

    def create_or_load(self, sess, out_dir_model, feed_dict=None):
        latest_ckpt = tf.train.get_checkpoint_state(out_dir_model)
        if latest_ckpt:
            path = latest_ckpt.model_checkpoint_path
            print('loading pre-trained model from %s.....' % path)
            self.saver.restore(sess, path)
        else:
            sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(self.input.initializer, feed_dict=feed_dict)

    def train(self, sess, hparams):
        self.create_or_load(sess, hparams.out_dir_model)
        current_step = sess.run(self.global_step)
        summary_writer = tf.summary.FileWriter(hparams.out_dir_summary)
        if current_step == 0:
            summary_writer.add_graph(tf.get_default_graph())
        while current_step < hparams.num_train_steps:
            _, loss, step_summary, current_step = sess.run(
                [self.train_op, self.loss, self.train_summary, self.global_step])
            if current_step % 10 == 0:
                self.save(sess, hparams.out_dir_model + '/model.ckpt', global_step=current_step)
                print("save train model step=%d path=%s" % (current_step, hparams.out_dir_model))
                # Write step summary.
                summary_writer.add_summary(step_summary, current_step)

    def validate(self, sess, hparams):
        self.create_or_load(sess, hparams.out_dir_model)
        while True:
            try:
                tf_loss, tf_accuracy = sess.run([self.loss, self.accuracy])
                print('loss=%s, accuracy=%s' % (tf_loss, tf_accuracy))
            except tf.errors.OutOfRangeError:
                print('validate finished!')
                break

    def predict(self, sess, hparams, feed_dict):
        # 获取原文本的iterator
        tag_table = lookup_ops.index_to_string_table_from_file(
            hparams.tgt_vocab_file, default_value='<tag-unknown>')
        self.create_or_load(sess, hparams.out_dir_model, feed_dict=feed_dict)
        tf_viterbi_sequence = sess.run(self.viterbi_sequence)[0]

        tags = []
        for id in tf_viterbi_sequence:
            tag = sess.run(tag_table.lookup(tf.constant(id, dtype=tf.int64)))
            tags.append(tag.decode('UTF-8'))
        return tags

    def save(self, sess, model_path, global_step):
        self.saver.save(sess, model_path, global_step)

    def _build_graph(self, hparams):
        # source = tf.Print(self.input.source, [self.input.source], message="source=", summarize=1000)
        # tgt = tf.Print(self.input.target_input, [self.input.target_input], message="tgt=", summarize=1000)
        wi = tf.nn.embedding_lookup(self.embedding, self.input.wi_ids)
        wj = tf.nn.embedding_lookup(self.embedding, self.input.wj_ids)
        ci = tf.one_hot(self.input.ci_ids, hparams.c_vocab_size)
        cj = tf.one_hot(self.input.cj_ids, hparams.c_vocab_size)
        tgt = self.input.deprel_ids

        # x: [batch_size, time_step, embedding_size], float32
        self.x = tf.concat([wi, wj, ci, cj], 2)
        # y: [batch_size, time_step]
        self.y = tgt

        cell_forward = self._single_cell(hparams)
        cell_backward = self._single_cell(hparams)

        # time_major 可以适应输入维度。
        outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x,
                                                            sequence_length=self.input.sequence_length,
                                                            dtype=tf.float32)
        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)
        # outputs = tf.Print(outputs, [outputs], summarize=10000)

        # projection:
        w = tf.get_variable("projection_w", [2 * hparams.num_units, hparams.tgt_vocab_size])
        b = tf.get_variable("projection_b", [hparams.tgt_vocab_size])
        x_reshape = tf.reshape(outputs, [-1, 2 * hparams.num_units], name="outputs_x_reshape")
        projection = tf.matmul(x_reshape, w) + b

        # -1 to time step
        max_time = outputs.shape[1].value or tf.shape(outputs)[1]
        self.outputs = tf.reshape(projection, [self.batch_size, max_time, hparams.tgt_vocab_size],
                                  name="outputs_reshape")

        num_tags = hparams.tgt_vocab_size
        self.transition_params = tf.get_variable("transitions", [num_tags, num_tags])
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.outputs, self.transition_params,
                                                                    self.input.sequence_length)
        self.viterbi_sequence = viterbi_sequence
        if hparams.action == 'train' or hparams.action == 'validate':
            with tf.name_scope("crf"):
                self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.outputs, self.y, self.input.sequence_length, transition_params=self.transition_params)
                # Add a training op to tune the parameters.
                self.loss = tf.reduce_mean(-self.log_likelihood)
                learning_rate = hparams.learning_rate
                global_step = self.global_step
                self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss,
                                                                                          global_step=global_step)

                correct_prediction = equal_float(self.y, viterbi_sequence)
                # target_weights = tf.sequence_mask(
                #     self.inputs.target_sequence_length, max_time, dtype=tf.float32)

                self.accuracy = tf.reduce_mean(tf.divide(tf.reduce_sum(correct_prediction, axis=1),
                                                         tf.cast(self.input.sequence_length, dtype=tf.float32)))
                self.train_summary = tf.summary.merge([
                    tf.summary.scalar("accuracy", self.accuracy),
                    tf.summary.scalar("train_loss", self.loss),
                ])

    @staticmethod
    def _single_cell(hparams):
        cell = tf.contrib.rnn.BasicLSTMCell(hparams.num_units, forget_bias=hparams.forget_bias)
        if hparams.dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - hparams.dropout))
        return cell
