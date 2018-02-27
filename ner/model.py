import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from ner.utils import file_content_iterator, equal_float


class NerModel(object):
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

    def create_or_load(self, sess, out_dir_model):
        latest_ckpt = tf.train.get_checkpoint_state(out_dir_model)
        if latest_ckpt:
            path = latest_ckpt.model_checkpoint_path
            print('loading pre-trained model from %s.....' % path)
            self.saver.restore(sess, path)
        else:
            sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(self.input.initializer)

    def train(self, sess, hparams):
        self.create_or_load(sess, hparams.out_dir_model)
        current_step = sess.run(self.global_step)
        summary_writer = tf.summary.FileWriter(hparams.out_dir_summary)
        if current_step == 0:
            summary_writer.add_graph(tf.get_default_graph())
        while current_step < hparams.num_train_steps:
            _, loss, step_summary, current_step = sess.run(
                [self.train_op, self.loss, self.train_summary, self.global_step])
            if current_step % 50 == 0:
                self.save(sess, hparams.out_dir_model + '/model.ckpt', global_step=current_step)
                print("save train model step=%d path=%s" % (current_step, hparams.out_dir_model))
                # Write step summary.
                summary_writer.add_summary(step_summary, current_step)

    def predict(self, sess, hparams):
        # 获取原文本的iterator
        file_iter = file_content_iterator(hparams.pred)
        tag_table = lookup_ops.index_to_string_table_from_file(
            hparams.tgt_vocab_file, default_value='<tag-unknown>')
        self.create_or_load(sess, hparams.out_dir_model)
        while True:
            try:
                tf_viterbi_sequence = sess.run(self.viterbi_sequence)[0]

            except tf.errors.OutOfRangeError:
                print('Prediction finished!')
                break

            tags = []
            for id in tf_viterbi_sequence:
                tags.append(sess.run(tag_table.lookup(tf.constant(id, dtype=tf.int64))))
            # write_result_to_file(file_iter, tags)
            raw_content = next(file_iter)
            words = raw_content.split(' ')
            assert len(words) == len(tags)
            for w, t in zip(words, tags):
                print(w, '(' + t.decode("UTF-8") + ')')
            print()
            print('*' * 100)

    def save(self, sess, model_path, global_step):
        self.saver.save(sess, model_path, global_step)

    def _build_graph(self, hparams):
        # source = tf.Print(self.input.source, [self.input.source], message="source=", summarize=1000)
        # tgt = tf.Print(self.input.target_input, [self.input.target_input], message="tgt=", summarize=1000)
        source = self.input.source
        tgt = self.input.target_input

        # x: [batch_size, time_step, embedding_size], float32
        self.x = tf.nn.embedding_lookup(self.embedding, source)
        # y: [batch_size, time_step]
        self.y = tgt

        cell_forward = self._single_cell(hparams)
        cell_backward = self._single_cell(hparams)

        # time_major 可以适应输入维度。
        outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x,
                                                            sequence_length=self.input.source_sequence_length,
                                                            dtype=tf.float32)
        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)
        # outputs = tf.Print(outputs, [outputs], summarize=10000)

        # projection:
        w = tf.get_variable("projection_w", [2 * hparams.num_units, hparams.class_size])
        b = tf.get_variable("projection_b", [hparams.class_size])
        x_reshape = tf.reshape(outputs, [-1, 2 * hparams.num_units], name="outputs_x_reshape")
        projection = tf.matmul(x_reshape, w) + b

        # -1 to time step
        max_time = outputs.shape[1].value or tf.shape(outputs)[1]
        self.outputs = tf.reshape(projection, [self.batch_size, max_time, hparams.class_size],
                                  name="outputs_reshape")

        num_tags = hparams.class_size
        self.transition_params = tf.get_variable("transitions", [num_tags, num_tags])
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.outputs, self.transition_params,
                                                                    self.input.source_sequence_length)
        self.viterbi_sequence = viterbi_sequence
        if hparams.action == 'train':
            with tf.name_scope("crf"):
                self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.outputs, self.y, self.input.source_sequence_length, transition_params=self.transition_params)
                # Add a training op to tune the parameters.
                self.loss = tf.reduce_mean(-self.log_likelihood)
                learning_rate = hparams.learning_rate
                global_step = self.global_step
                # params = tf.trainable_variables()
                # dency_step = hparams.learning_rate_tau_steps
                # alpha = tf.to_float(tf.divide(self.global_step, dency_step))

                # learning_rate = tf.cond(global_step < dency_step,
                #                         lambda: tf.to_float(
                #                             (1.0 - alpha) * hparams.learning_rate_tau_factor * learning_rate
                #                             + alpha * learning_rate),
                #                         lambda: learning_rate)
                # learning_rate = tf.cond(global_step > hparams.start_decay_step,
                #                         lambda: tf.train.exponential_decay(learning_rate,
                #                                                            tf.subtract(global_step, hparams.start_decay_step),
                #                                                            hparams.decay_steps, hparams.decay_factor),
                #                         lambda: learning_rate)
                # gradients = tf.gradients(self.loss, params, name="gradients")
                # clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)
                # gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm),
                #                          tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))]
                # opt = tf.train.GradientDescentOptimizer(learning_rate)
                #
                # self.train_op = opt.apply_gradients(
                #     zip(clipped_gradients, params), global_step=global_step)
                self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss,
                                                                                          global_step=global_step)

                correct_prediction = equal_float(self.y, viterbi_sequence)
                # target_weights = tf.sequence_mask(
                #     self.inputs.target_sequence_length, max_time, dtype=tf.float32)

                self.accuracy = tf.reduce_mean(tf.divide(tf.reduce_sum(correct_prediction, axis=1),
                                                         tf.cast(self.input.source_sequence_length, dtype=tf.float32)))
                self.train_summary = tf.summary.merge([
                    tf.summary.scalar("accuracy", self.accuracy),
                    tf.summary.scalar("train_loss", self.loss),
                ])

    def _single_cell(self, hparams):
        cell = tf.contrib.rnn.BasicLSTMCell(hparams.num_units, forget_bias=hparams.forget_bias)
        if hparams.dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - hparams.dropout))
        return cell
