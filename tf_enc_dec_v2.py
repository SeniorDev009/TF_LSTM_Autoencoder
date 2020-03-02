import tensorflow.compat.v1 as tf



class LSTMAutoencoderV2():
    def __init__(self, batchsize, window_width, encoder_layer_list, decoder_layer_list, num_class=1):
        self.window_width = window_width
        self.encoder_layer_list = encoder_layer_list
        self.decoder_layer_list = decoder_layer_list
        self.num_class = num_class
        self.batchsize = batchsize
        self.decode_without_input = False
        self.optimizer = None
        self.reverse = None
        self.output_enc = None
        self.output_dec = None

        self.lstm_cell_enc = [tf.nn.rnn_cell.LSTMCell(size) for size in encoder_layer_list]
        self.multi_rnn_enc_cell = tf.nn.rnn_cell.MultiRNNCell(self.lstm_cell_enc)
        self.initial_state_enc = self.multi_rnn_enc_cell.zero_state(self.batchsize, dtype=tf.float32)

        self.lstm_cell_dec = [tf.nn.rnn_cell.LSTMCell(size) for size in decoder_layer_list]
        self.multi_rnn_dec_cell = tf.nn.rnn_cell.MultiRNNCell(self.lstm_cell_dec)
        self.initial_state_dec = self.multi_rnn_dec_cell.zero_state(self.batchsize, dtype=tf.float32)

        self.x_placeholder = tf.placeholder(tf.float32, [self.batchsize, self.window_width, self.num_class])
        self.y_placeholder = tf.placeholder(tf.float32, [self.batchsize, self.window_width, self.num_class])

        self.encoder()
        self.decoder()

    def encoder(self):
        with tf.variable_scope('encoder') as encoder:
            self.output_enc, self.initial_state_enc = tf.nn.dynamic_rnn(cell=self.multi_rnn_enc_cell,
                                                                        inputs=self.x_placeholder,
                                                                        initial_state=self.initial_state_enc,
                                                                        dtype=tf.float32, time_major=False)

        # fc_output = tf.layer

    def decoder(self):
        with tf.variable_scope('decoder') as decoder:


            if self.decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(self.batchsize, self.window_width, self.num_class),
                                       dtype=tf.float32)]

                self.output_dec, self.initial_state_dec = tf.nn.dynamic_rnn(cell=self.multi_rnn_dec_cell,
                                                                            inputs=dec_inputs,
                                                                            initial_state=self.initial_state_enc,
                                                                            dtype=tf.float32, time_major=False)

                if self.reverse:
                    self.output_dec = self.output_dec[::-1]
            else:

                self.output_dec, self.initial_state_dec = tf.nn.dynamic_rnn(cell=self.multi_rnn_dec_cell,
                                                                            inputs= self.output_enc,
                                                                            initial_state=self.initial_state_enc,
                                                                            dtype=tf.float32, time_major=False)
            self.output_dense_dec = tf.layers.dense(self.output_dec, self.num_class, name="output")

            self.loss = tf.reduce_mean(tf.square(self.y_placeholder - self.output_dense_dec))

            if self.optimizer is None:
                self.train = tf.train.AdamOptimizer().minimize(self.loss)
            else:
                self.train = self.optimizer.minimize(self.loss)

            tf.summary.scalar("mini_batch_loss", self.loss)
            self.merged = tf.summary.merge_all()


