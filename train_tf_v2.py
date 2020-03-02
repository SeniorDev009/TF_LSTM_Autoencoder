from __future__ import absolute_import, division, print_function, unicode_literals
import librosa

from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
import tensorflow.compat.v1 as tf
import numpy as np
import os
from tf_enc_dec_v2 import LSTMAutoencoderV2
import aux_fn
import logging
import time

from tensorflow.contrib import cudnn_rnn
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

#parameters
batchsize = 100
epochs = 100
sample_rate = 16000
window_width = 16000 #this is size of window to put into autoencoder.

logs_path = "results/log"
model_path = "results/model"
path = 'test_data/audio'
encoder_layer_list = [15 , 15, 15, 15]
decoder_layer_list = [15, 15, 15, 15]
if not os.path.exists(logs_path):
    os.mkdir(logs_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename="results/training.log",filemode='w', level=logging.INFO)



class LogHistory(Callback):
    def on_epoch_end(self, batch, logs={}):
        logging.info(logs)

model = LSTMAutoencoderV2(batchsize, window_width, encoder_layer_list, decoder_layer_list)

files = aux_fn.get_all_files(path)
print(['Number of files',len(files)])

data = aux_fn.get_data_from_files(files, sr=sample_rate, ww=window_width)
data = np.expand_dims(data, axis=2)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    train_writer = tf.summary.FileWriter(logs_path, sess.graph)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        minibatch = 0
        iteration = int((data.shape[0]) / batchsize)
        for i in range(iteration):
            start = minibatch
            end = minibatch + batchsize
            batch_data = np.array(data[start:end, :, :])
            (loss_val, optimizer, merged) = sess.run([model.loss, model.train, model.merged], {model.x_placeholder: batch_data, model.y_placeholder: batch_data })
            print('iter %d:' % (i + 1), loss_val)
            epoch_loss += loss_val
            minibatch += batchsize
            tf.summary.scalar("mini_batch_loss", loss_val)
            train_writer.add_summary(merged, i)



        batch_data = np.array(data[start:end, :, :])
        print('Epoch ', epoch, 'Completed out of', epochs, "loss: ", epoch_loss)
        (output) = sess.run([model.output_dense_dec], {model.x_placeholder: batch_data})
        output = list(output)
        output = output[0]
        print('train result :')
        print('input :', batch_data[0, :, :].flatten())
        print('output :', output[0, :, :].flatten())
        #summary_lossepoch = tf.summary.scalar("epoch_loss", epoch_loss)
        #summary_lossepochavg = tf.summary.scalar("epoch_loss", epoch_loss)
        #tf.summary.scalar("avg_epoch_loss", epoch_loss / int((data.shape[0]) / batchsize) )
        #merged = tf.summary.merge_all()
        #train_writer.add_summary(summary_lossepoch, epoch)
        #train_writer.add_summary(summary_lossepochavg, epoch)


    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    tf.io.write_graph(sess.graph_def, './results', 'network_trainV2.pbtxt')
    output_node_names = "decoder/dense_output"
    output_graph = os.path.join(model_path, "output_tfgraphv2.pb")
    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(",")
                                                                 )
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    graph_io.write_graph(output_graph_def, model_path, 'output_tfgraphv2.pbtxt', as_text=True)


