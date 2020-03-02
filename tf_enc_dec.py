import tensorflow.compat.v1 as tf
from tensorflow.contrib import rnn
from tensorflow.python.framework import graph_util


class LSTMAutoencoder():
    def __init__(self, batchsize, window_width, LAYER_SIZE, HIDDEN_LAYER_COUNT=0):
        self.window_width = window_width
        self.LAYER_SIZE = LAYER_SIZE
        self.HIDDEN_LAYER_COUNT = HIDDEN_LAYER_COUNT
        self.batchsize = batchsize
        self.decode_without_input = False
        self.optimizer = None
        self.reverse = None
        self.output_ = None

        self.lstm_cell_enc = rnn.BasicLSTMCell(self.LAYER_SIZE)
        if self.HIDDEN_LAYER_COUNT > 0:
            self.lstm_cell_enc = rnn.MultiRNNCell(
                [rnn.BasicLSTMCell(self.LAYER_SIZE) for _ in range(self.HIDDEN_LAYER_COUNT + 1)])
        self.initial_state_enc = self.lstm_cell_enc.zero_state(self.batchsize, dtype=tf.float32)
        self.lstm_cell_dec = rnn.BasicLSTMCell(self.LAYER_SIZE)
        if self.HIDDEN_LAYER_COUNT > 0:
            self.lstm_cell_dec = rnn.MultiRNNCell(
                [rnn.BasicLSTMCell(self.LAYER_SIZE) for _ in range(self.HIDDEN_LAYER_COUNT + 1)])
        self.initial_state_dec = self.lstm_cell_dec.zero_state(self.batchsize, dtype=tf.float32)

        self.x_placeholder = tf.placeholder(tf.float32, [self.batchsize, self.window_width])
        self.y_placeholder = tf.placeholder(tf.float32, [self.batchsize, self.window_width])

        self.encoder()
        self.decoder()

    def encoder(self):
        self.inputdata = tf.split(self.x_placeholder, self.window_width, 1)
        (self.z_codes, self.enc_state) = rnn.static_rnn(self.lstm_cell_enc, self.inputdata, initial_state=self.initial_state_enc, dtype=tf.float32)
        # fc_output = tf.layer

    def decoder(self):
        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.truncated_normal([self.LAYER_SIZE,
                                                           1], dtype=tf.float32), name='dec_weight'
                                      )
            dec_bias_ = tf.Variable(tf.constant(0.1,
                                                shape=[1],
                                                dtype=tf.float32), name='dec_bias')

            if self.decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(self.batchsize, self.window_width),
                                       dtype=tf.float32)]
                (dec_outputs, dec_state) = rnn.static_rnn(self.lstm_cell_dec, dec_inputs,
                                                          initial_state=self.enc_state,
                                                          dtype=tf.float32)
                if self.reverse:
                    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0,
                                                                   2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0),
                                      [self.batchsize, 1, 1])
                self.output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_
            else:

                (dec_outputs, dec_state) = rnn.static_rnn(self.lstm_cell_dec, self.z_codes,
                                                          initial_state=self.enc_state,
                                                          dtype=tf.float32)

                if self.reverse:
                    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0,
                                                                   2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0),
                                      [self.batchsize, 1, 1])
                self.output_ = tf.add(tf.matmul(dec_output_, dec_weight_) , dec_bias_ , name="dense_output")

                # dec_state = self.enc_state
                # dec_input_ = tf.zeros(tf.shape(self.batchsize, self.window_width),
                #                       dtype=tf.float32)
                # dec_outputs = []
                # for step in range(len(self.x_placeholder)):
                #     if step > 0:
                #         vs.reuse_variables()
                #     (dec_input_, dec_state) = self.lstm_cell_dec(dec_input_, dec_state)
                #     dec_input_ = tf.matmul(dec_input_, dec_weight_)  + dec_bias_
                #     dec_outputs.append(dec_input_)
                # if self.reverse:
                #     dec_outputs = dec_outputs[::-1]
                # self.output_ = tf.transpose(tf.stack(dec_outputs), [1,
                #                                                     0, 2])

            # self.input_ = tf.transpose(tf.stack(self.x_placeholder), [1, 0, 2])
            # self.input_ = tf.transpose(tf.stack(self.x_placeholder), [1, 0])
            # self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_ ))
            self.input_ = tf.stack(tf.expand_dims(self.x_placeholder , 2))
            self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))

            if self.optimizer is None:
                self.train = tf.train.AdamOptimizer().minimize(self.loss)
            else:
                self.train = self.optimizer.minimize(self.loss)

            tf.summary.scalar("mini_batch_loss", self.loss)
            self.merged = tf.summary.merge_all()


