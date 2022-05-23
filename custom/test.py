from mpi4py import MPI
import pandas as pd
import numpy as np
import os, argparse
import cv2
import sys
import tensorflow as tf
import datetime
import time

t1 = time.time()

sys.path.insert(1, '/p/project/joaiml/ingolfsson1/COVID_Net')
from data import process_image_file

comm = MPI.COMM_WORLD

def inv_mapping(a):
    if a['pred'] == 0:
        return 'normal'
    elif a['pred'] == 1:
        return 'pneumonia'
    else:
        return 'COVID-19'


def predict(imagepath, weightspath, metaname, ckptname, in_tensorname, out_tensorname, input_size):

    top_percent = 0.08

    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))
    saver.restore(sess, os.path.join(weightspath, ckptname))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(in_tensorname)
    pred_tensor = graph.get_tensor_by_name(out_tensorname)

    x = process_image_file(imagepath, top_percent, input_size)
    x = x.astype('float32') / 255.0
    pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

    #return inv_mapping[pred.argmax(axis=1)[0]]
    return pred.argmax(axis=1)[0]

"""
weightspath = '/p/project/joaiml/ingolfsson1/COVID_Net/models/COVIDNet-CXR4-A'
metaname = 'model.meta'
ckptname = 'model-18540'
in_tensorname = 'input_1:0'
out_tensorname = 'norm_dense_1/Softmax:0'
input_size = 480
split = 'test'
"""

weightspath = '/p/project/joaiml/ingolfsson1/COVID_Net/models/COVIDNet-CXR-Large'
metaname = 'model.meta'
ckptname = 'model-8485'
in_tensorname = 'input_1:0'
out_tensorname = 'dense_3/Softmax:0'
input_size = 224
split = 'test'

tf.logging.set_verbosity(tf.logging.ERROR)

rank = comm.Get_rank()
size = comm.Get_size()

df = pd.read_csv('/p/project/joaiml/ingolfsson1/COVID_Net/results/covidx_test_base.csv')
#df = pd.read_csv('/p/project/joaiml/ingolfsson1/COVID_Net/results/total_ehl_base.csv')

if rank == 0:
    sendbuf = np.array_split(df.index.values, size)
else:
    sendbuf = None

recvbuf = comm.scatter(sendbuf, root=0)

data = np.empty([0, 3])

for i in recvbuf:
    path = df.iloc[i]['path']
    #split = df.iloc[i]['split']
    complete_path = '/p/project/joaiml/ingolfsson1/COVID_Net/data/{}/{}'.format(split, path)
    t = time.time()
    pred = predict(complete_path, weightspath, metaname, ckptname, in_tensorname, out_tensorname, input_size)   
    t = time.time() - t
    data = np.concatenate((data, np.array([[i, pred, t]])), axis=0)
    print("Rank {} - Image {} - Prediction {} - Time {}".format(rank, path, pred, t))

# path = df.iloc[rank]['path']
# complete_path = '/p/project/joaiml/ingolfsson1/COVID_Net/data/{}/{}'.format(split, path)
# pred = predict(complete_path, weightspath, metaname, ckptname, in_tensorname, out_tensorname, input_size)   
# data = np.concatenate((data, np.array([[rank, pred]])), axis=0)
# print("Rank {} - Image {} - Prediction {}".format(rank, path, pred))

data = comm.gather(data, root=0)

if rank == 0:
    print("Rank {} - gathering data of length: {}".format(rank, len(data)))
    #print("")
    #print(np.concatenate(data))
    df_data = pd.DataFrame(np.concatenate(data), columns=['index', 'pred', 'time'])
    df_out = df[['id', 'path', 'condition', 'source']].merge(df_data, left_index=True, right_on='index')
    df_out = df_out.drop('index', axis=1)
    df_out['pred'] = df_out.apply(inv_mapping, axis=1)
    t1 = time.time() - t1
    df_out.to_csv('/p/project/joaiml/ingolfsson1/COVID_Net/results/covidx_test_{}_224_{}.csv'.format(size, t1), index=False)
    print('Created csv file: /p/project/joaiml/ingolfsson1/COVID_Net/results/covidx_test_{}_224_{}.csv'.format(size, int(t1)))
    print("Total inference time: {}".format(t1))