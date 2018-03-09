import tensorflow as tf
from tensorflow.python.ops import lookup_ops


class ClassifyModel(object):
    def __init__(self, hparams, input, embedding, batch_size=None, mode=tf.contrib.learn.ModeKeys.INFER) -> None:
        self.input = input
        self.mode = mode
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

    def train(self, sess, hparams):
        sess.run(self.input.initializer)
        current_step = sess.run(self.global_step)
        summary_writer = tf.summary.FileWriter(hparams.out_dir_summary)
        if current_step == 0:
            summary_writer.add_graph(tf.get_default_graph())
            tf_source, tf_target = sess.run([self.input.source, self.input.target])
            print(tf_source)
            print(tf_target)
        while current_step < hparams.num_train_steps:

            _, loss, step_summary, current_step = sess.run(
                [self.train_op, self.loss, self.train_summary, self.global_step])
            if current_step % 10 == 0:
                self.save(sess, hparams.out_dir_model + '/model.ckpt', global_step=current_step)
                print("save train model step=%d path=%s" % (current_step, hparams.out_dir_model))
                # Write step summary.
                summary_writer.add_summary(step_summary, current_step)

    def train_online(self, sess, hparams, num_train_steps, feed_dict=None):
        sess.run(self.input.initializer, feed_dict=feed_dict)
        summary_writer = tf.summary.FileWriter(hparams.out_dir_summary)
        for _ in range(num_train_steps):
            _, loss, step_summary, current_step = sess.run(
                [self.train_op, self.loss, self.train_summary, self.global_step])
            self.save(sess, hparams.out_dir_model + '/model.ckpt', global_step=current_step)
            print("save train model step=%d path=%s" % (current_step, hparams.out_dir_model))
            summary_writer.add_summary(step_summary, current_step)

    def predict(self, sess, feed_dict=None):
        sess.run(self.input.initializer, feed_dict=feed_dict)
        # 获取原文本的iterator
        tf_tag = sess.run([self.tag])
        return tf_tag[0].decode('utf-8')

    def save(self, sess, model_path, global_step):
        self.saver.save(sess, model_path, global_step)

    def _build_graph(self, hparams):
        source = self.input.source
        tgt = self.input.target
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
        forward_state, backward_state = bi_state
        bi_state = tf.concat([forward_state.h, backward_state.h], axis=1)

        # projection:
        w = tf.get_variable("projection_w", [2 * hparams.num_units, hparams.class_size])
        b = tf.get_variable("projection_b", [hparams.class_size])
        # x_reshape = tf.reshape(outputs, [-1, 2 * hparams.num_units], name="outputs_x_reshape")
        self.outputs = tf.matmul(bi_state, w) + b

        num_tags = hparams.class_size
        self.transition_params = tf.get_variable("transitions", [num_tags, num_tags])
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.y))
            # Add a training op to tune the parameters.
            learning_rate = hparams.learning_rate
            global_step = self.global_step
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss,
                                                                                      global_step=global_step)

            correct_prediction = tf.equal(tf.cast(self.y, dtype=tf.int64), tf.argmax(self.outputs, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_summary = tf.summary.merge([
                tf.summary.scalar("accuracy", self.accuracy),
                tf.summary.scalar("train_loss", self.loss),
            ])
        else:
            self.projection = tf.nn.softmax(self.outputs)
            self.projection_id = tf.argmax(self.outputs, axis=1)
            tag_table = lookup_ops.index_to_string_table_from_file(
                hparams.tgt_vocab_file, default_value='<unk>')
            self.tag = tag_table.lookup(self.projection_id)

    def _single_cell(self, hparams):
        cell = tf.contrib.rnn.BasicLSTMCell(hparams.num_units, forget_bias=hparams.forget_bias)
        if hparams.dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - hparams.dropout))
        return cell
