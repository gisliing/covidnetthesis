from __future__ import print_function
from mpi4py import MPI
import tensorflow as tf
import os, argparse, pathlib
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(1, '/p/project/joaiml/ingolfsson1/COVID_Net')

from eval import eval
from data import BalanceCovidDataset

comm = MPI.COMM_WORLD

tf.logging.set_verbosity(tf.logging.ERROR)

rank = comm.Get_rank()
size = comm.Get_size()

jobs = pd.read_csv('/p/home/jusers/ingolfsson1/deep/COVID-Net/training_joblist.csv')

if rank == 0:
    sendbuf = np.array_split(jobs.index.values, size)
else:
    sendbuf = None

recvbuf = comm.scatter(sendbuf, root=0)

data = np.empty([0, 3])

def covidnet_train(epochs, learning_rate, batch_size, weightspath, metaname, ckptname,trainfile,
                   testfile, class_weights, covid_percent, input_size, out_tensorname, in_tensorname,
                   logit_tensorname, label_tensorname, weights_tensorname):
    display_step = 1
    outputPath = '/p/project/joaiml/ingolfsson1/COVID_Net/output/'
    runID = 'COVIDNet-lr' + str(learning_rate)
    runPath = outputPath + runID
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
    print('Output: ' + runPath)

    with open(trainfile) as f:
        trainfiles = f.readlines()
    with open(testfile) as f:
        testfiles = f.readlines()

    generator = BalanceCovidDataset(data_dir='/p/project/joaiml/ingolfsson1/COVID_Net/data',
                                    csv_file=trainfile,
                                    covid_percent=covid_percent,
                                    class_weights=class_weights)

    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.get_default_graph()
        saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))

        graph = tf.get_default_graph()

        image_tensor = graph.get_tensor_by_name(in_tensorname)
        labels_tensor = graph.get_tensor_by_name(label_tensorname)
        sample_weights = graph.get_tensor_by_name(weights_tensorname)
        pred_tensor = graph.get_tensor_by_name(logit_tensorname)
        # loss expects unscaled logits since it performs a softmax on logits internally for efficiency

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred_tensor, labels=labels_tensor)*sample_weights)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Initialize the variables
        init = tf.global_variables_initializer()

        # Run the initializer
        sess.run(init)

        # load weights
        saver.restore(sess, os.path.join(weightspath, ckptname))
        #saver.restore(sess, tf.train.latest_checkpoint(args.weightspath))

        # save base model
        saver.save(sess, os.path.join(runPath, 'model'))
        print('Saved baseline checkpoint')
        print('Baseline eval:')
        eval(sess, graph, testfiles, os.path.join('/p/project/joaiml/ingolfsson1/COVID_Net/data','test'))

        # Training cycle
        print('Training started')
        total_batch = len(generator)
        progbar = tf.keras.utils.Progbar(total_batch)
        
        losses = []
        
        for epoch in range(epochs):
            for i in range(total_batch):
                # Run optimization
                batch_x, batch_y, weights = next(generator)

                sess.run(train_op, feed_dict={image_tensor: batch_x,
                                              labels_tensor: batch_y,
                                              sample_weights: weights})
                progbar.update(i+1)

            if epoch % display_step == 0:
                pred = sess.run(pred_tensor, feed_dict={image_tensor:batch_x})
                loss = sess.run(loss_op, feed_dict={pred_tensor: pred,
                                                    labels_tensor: batch_y,
                                                    sample_weights: weights})
                print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
                losses.append(loss)
                eval(sess, graph, testfiles, os.path.join('/p/project/joaiml/ingolfsson1/COVID_Net/data','test'))
                saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch+1, write_meta_graph=False)
                print('Saving checkpoint at epoch {}'.format(epoch + 1))

    print("Optimization Finished!")
    print("Rank {}: Losses: {}".format(rank, losses))
    return loss


for i in recvbuf:
    print("{}/{} working on job {}".format(rank, size-1, i))
    
    run = jobs.iloc[i]

    epochs = 25
    learning_rate = run['learning_rate']
    batch_size = run['batch_size']
    weightspath = '/p/project/joaiml/ingolfsson1/COVID_Net/{}'.format(run['model'])
    metaname = 'model.meta'
    ckptname = 'model-8485'
    n_classes = 3
    trainfile = '/p/project/joaiml/ingolfsson1/COVID_Net/labels/train_{}.txt'.format(run['labels'])
    testfile = '/p/project/joaiml/ingolfsson1/COVID_Net/labels/test_{}.txt'.format(run['labels'])
    class_weights = [run['classweights_normal'], run['classweights_pneumonia'], run['classweights_covid19']]
    covid_percent = run['covid_percent']
    input_size = 224
    out_tensorname = 'dense_3/Softmax:0'
    in_tensorname = 'input_1:0'
    logit_tensorname = 'dense_3/MatMul:0'
    label_tensorname = 'dense_3_target:0'
    weights_tensorname = 'dense_3_sample_weights:0'

    t = time.time()
    loss = covidnet_train(epochs, learning_rate, batch_size, weightspath, metaname, ckptname, trainfile, testfile, class_weights, covid_percent, input_size, out_tensorname, in_tensorname, logit_tensorname, label_tensorname, weights_tensorname)
    t = time.time() - t
    data = np.concatenate((data, np.array([[i, loss, t]])), axis=0)

data = comm.gather(data, root=0)

if rank == 0:
    print("{}/{} gathering data of len {}".format(rank, size-1, len(data)))
    df_data = pd.DataFrame(np.concatenate(data), columns=['index', 'loss', 'time'])
    df_out = jobs.merge(df_data, left_index=True, right_on='index')
    df_out = df_out.drop('index', axis=1)
    df_out.to_csv('/p/home/jusers/ingolfsson1/deep/COVID-Net/training_runs.csv', index=False)
    print("All jobs finished")
