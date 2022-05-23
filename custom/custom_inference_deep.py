import subprocess
import time
import pandas as pd
import numpy as np
import datetime
import tensorflow
import os, argparse
import cv2
import sys

#from p.project.joaiml.ingolfsson1.COVID_Net.data import process_image_file

#os.chdir("/p/project/joaiml/ingolfsson1/COVID_Net")

sys.path.insert(1, '/p/project/joaiml/ingolfsson1/COVID_Net')

from data import process_image_file

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df_test = pd.read_csv('/p/project/joaiml/ingolfsson1/COVID_Net/labels/total_ehl.txt', sep=' ', header=None)
df_test.columns = ["id", "path", "condition", "source", "split"]

df_out = pd.read_csv('/p/project/joaiml/ingolfsson1/COVID_Net/results/total_ehl_1633595102.csv', sep=',')

def format_pred(out):
    return out.decode('utf-8').split('\n')[0].split(' ')[-1]

def predict(imagepath, split):
    
    weightspath = '/p/project/joaiml/ingolfsson1/COVID_Net/models/COVIDNet-CXR4-A'
    metaname = 'model.meta'
    ckptname = 'model-18540'
    in_tensorname = 'input_1:0'
    out_tensorname = 'norm_dense_1/Softmax:0'
    input_size = 480
    top_percent = 0.08
    
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
    
    tensorflow.reset_default_graph()
    sess = tensorflow.Session()
    tensorflow.get_default_graph()
    saver = tensorflow.train.import_meta_graph(os.path.join(weightspath, metaname))
    saver.restore(sess, os.path.join(weightspath, ckptname))
    
    graph = tensorflow.get_default_graph()

    image_tensor = graph.get_tensor_by_name(in_tensorname)
    pred_tensor = graph.get_tensor_by_name(out_tensorname)

    #print('p/project/joaiml/ingolfsson1/COVID_Net/data/{}/{}'.format(split, imagepath))
    
    x = process_image_file('/p/project/joaiml/ingolfsson1/COVID_Net/data/{}/{}'.format(split, imagepath), top_percent, input_size)
    x = x.astype('float32') / 255.0
    pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})
    
    prediction = inv_mapping[pred.argmax(axis=1)[0]]
    
    return prediction


total = 0
ts = int(datetime.datetime.now().timestamp())

for _, image, cond, source, split in df_test.values:
    
    if df_out[df_out['path'] == image]['pred'].iloc[0] == 'unknown':
    
        t = time.time()
        pred = predict(image, split)
        t = t - time.time()
        
        total = total + t

        print('Correct: {}'.format(cond))
        print('Predicted: {}'.format(pred))
        print('Inference time: {}'.format(t))
        print('Total time elapsed: {}'.format(total))
        print('')
    
        df_out['pred'] = np.where(df_out['path'] == image, pred, df_out['pred'])
        df_out['time'] = np.where(df_out['path'] == image, t, df_out['time'])
        
        df_out.to_csv('/p/project/joaiml/ingolfsson1/COVID_Net/results/total_ehl_{}.csv'.format(ts), index=False)

print('Inference complete.')
print('Total time elapsed: {}'.format(total))
