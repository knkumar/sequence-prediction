import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd
import glob
filenames = glob.glob('/Users/kitoo/Box Sync/Research/data_summer_2017/txtData/*.txt')


def read_and_decode(filename_queue):
    """
    Function to read and decode TF records for further processsing - 
        Read serialized file TF record and cast features variables to the right format
        This has to be changed for a new input pipeline.

    input  : filename_queue - the string input producer queue ingesting TF records
    output : variables to feed 

    """

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'Subject' : tf.FixedLenFeature([1],tf.int64),   # 1
        'period'  : tf.FixedLenFeature([1],tf.int64),   # 2
        'block'   : tf.FixedLenFeature([1],tf.int64),   # 3
        'stim'    : tf.FixedLenFeature([1],tf.int64),   # 4
        'foilInd' : tf.FixedLenFeature([1],tf.int64),   # 5
        'pos'     : tf.FixedLenFeature([1],tf.int64),   # 6
        'trial_id': tf.FixedLenFeature([1],tf.int64),   # 7
        'x_ord'   : tf.FixedLenFeature([1],tf.int64),   # 8
        'y_ord'   : tf.FixedLenFeature([1],tf.int64),   # 9
        'x_vel'   : tf.FixedLenFeature([1],tf.float32),   # 10
        'y_vel'   : tf.FixedLenFeature([1],tf.float32),   # 11
        'x_acc'    : tf.FixedLenFeature([1],tf.float32),   # 12 
        'y_acc'    : tf.FixedLenFeature([1],tf.float32),   # 13
        'out_x'   : tf.FixedLenFeature([1],tf.int64),     # 14
        'out_y'   : tf.FixedLenFeature([1],tf.int64),     # 15
        'out_xvel' : tf.FixedLenFeature([1],tf.float32),   # 16
        'out_yvel' : tf.FixedLenFeature([1],tf.float32),   # 17
        'out_xacc' : tf.FixedLenFeature([1],tf.float32),   # 18
        'out_yacc' : tf.FixedLenFeature([1],tf.float32),   # 19
        'time_after_stim' : tf.FixedLenFeature([1],tf.int64),     # 20
        })
    
    # Casting serialized string to the right format
    subject = tf.cast(features['Subject'], tf.int32)
    period = tf.cast(features['period'], tf.int32)
    block = tf.cast(features['block'], tf.int32)
    stim = tf.cast(features['stim'], tf.int32)
    foilInd = tf.cast(features['foilInd'], tf.int32)
    pos = tf.cast(features['pos'], tf.int32)
    trial_id = tf.cast(features['trial_id'], tf.int32)
    x_ord = tf.cast(features['x_ord'], tf.float32)
    y_ord = tf.cast(features['y_ord'], tf.float32)
    x_vel = tf.cast(features['x_vel'], tf.float32)
    y_vel = tf.cast(features['y_vel'], tf.float32)
    out_x = tf.cast(features['out_x'], tf.float32)
    out_y = tf.cast(features['out_y'], tf.float32)
    out_xvel = tf.cast(features['out_xvel'], tf.float32)
    out_yvel = tf.cast(features['out_yvel'], tf.float32)
    x_acc = tf.cast(features['x_acc'], tf.float32)
    y_acc = tf.cast(features['y_acc'], tf.float32)
    out_xacc = tf.cast(features['out_xacc'], tf.float32)
    out_yacc = tf.cast(features['out_yacc'], tf.float32)
    time_after_stim = tf.cast(features['time_after_stim'], tf.int32)

    # Rehspaing variable after cast
    subject = tf.reshape(subject, [1])
    period = tf.reshape(period, [1])
    block = tf.reshape(block, [1])
    stim = tf.reshape(stim, [1])
    foilInd = tf.reshape(foilInd, [1])
    pos = tf.reshape(pos, [1])
    trial_id = tf.reshape(trial_id, [1])
    x_ord = tf.reshape(x_ord, [1])
    y_ord = tf.reshape(y_ord, [1])
    x_vel = tf.reshape(x_vel, [1])
    y_vel = tf.reshape(y_vel, [1])
    out_x = tf.reshape(out_x, [1])
    out_y = tf.reshape(out_y, [1])
    out_xvel = tf.reshape(out_xvel, [1])
    out_yvel = tf.reshape(out_yvel, [1])
    x_acc = tf.reshape(x_acc, [1])
    y_acc = tf.reshape(y_acc, [1])
    out_xacc = tf.reshape(out_xacc, [1])
    out_yacc = tf.reshape(out_yacc, [1])
    time_after_stim = tf.reshape(time_after_stim, [1])
    
    return x_ord,y_ord,x_vel,y_vel,x_acc,y_acc,out_x,out_y,out_xvel,out_yvel, out_xacc, out_yacc, time_after_stim


def model(X, a, a_max, evidence, F, G, a_prev, evidence_dist, time_after_stim, delay_var, new_evidence):
    """
    A state space model designed to predict a future state based on the current state. This is an adaptation of the kalman filter for trajectory predictions.
    
    input : X - input vector
            a - control vector
            a_max - threshold for maximum control
            F - Transition matrix for Kalman Filter
            G - Control matrix
            evidence_dist - a distribution to add noise to the process (a normal distribution here)
            time_after_stim - timing index of evenets
            new_evidence - the accumulated control vector and noise from previous measurements

    output : X_hat - prediction for the next time point
             new_evidence - accumulated control vector and noise returned from current measurements
             accumulated_evidence - accumulated control and noise returned for later processing
    """

    # a_prev = tf.stop_gradient(a_prev)
    print("Running Model")
    if time_after_stim == delay_var:
        new_evidence = 0
    else:
        new_evidence = tf.add(new_evidence, tf.add(evidence, evidence_dist.sample([1])))

    sign_multiplier = tf.sign(a)
    x = tf.abs(a)

    accumulated_evidence = tf.subtract(1.0 ,  tf.exp( tf.negative( tf.multiply(x,new_evidence) ) ) )

    a_evidence = tf.add(a_prev , tf.multiply( tf.multiply(sign_multiplier,a_max), accumulated_evidence, name='aev'))

    X_hat = tf.add(tf.matmul(X,F), tf.matmul(a_evidence, tf.transpose(G)), name='calculate_xhat')
    # changed new_evidence to evidence to test the variability of accumulation
    return X_hat, new_evidence, accumulated_evidence


from tensorflow.python.framework import ops

def training(filenames_train):
    """
    function to train the model parameters using ADAM optimizer to calculate parameters minimizing prediction error

    input : filenames_train - filename of TF Record

    output : all_loss_values
    """

    # reset the graph for each subject
    ops.reset_default_graph()

    # create a string input producer to read the tfrecord
    filename_queue = tf.train.string_input_producer([filenames_train])
    # use read_and decode to retrieve tensors after casting and reshaping
    x_ord,y_ord,x_vel,y_vel,x_acc,y_acc,out_x,out_y,out_xvel,\
        out_yvel,out_xacc, out_yacc,time_after_stim = read_and_decode(filename_queue)  

    # create a batch to train the model
    batch = tf.train.batch([x_ord,y_ord,x_vel,y_vel,x_acc,y_acc,
                            out_x,out_y,out_xvel,out_yvel,out_xacc,out_yacc, time_after_stim], 
                           batch_size=1, capacity=2000, num_threads=1)

    # variables to feed from queue
    # input vector 
    X = tf.placeholder(tf.float32, shape=[1,4], name='input_vector') 
    # control vector
    a = tf.placeholder(tf.float32, shape=[1,2], name='a')
    # Prediction vector to compare loss
    X_pred = tf.placeholder(tf.float32, shape=[1, 4], name='pred_vector')
    # Previous control vector to update model state effectively
    a_prev = tf.placeholder(tf.float32, shape=[1,2], name='a_prev')
    # A time based interval when stimulus comes on
    timer_stim = tf.placeholder(tf.int32, shape=[1,1], name='timer_stim')



    # ------------- Model Parmeters -------------
    # initialization for transition matrix F
    f_init = tf.constant([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=tf.float32)
    # transition matrix F
    F = tf.get_variable('F', dtype=tf.float32, initializer=f_init)

    # initialization for control vector G
    g_init = tf.constant([[1,0],[0,1],[1,0],[0,1]], dtype=tf.float32)
    # control vector G which controls accumulated evidence affecting acceleration 
    G = tf.get_variable('G', dtype=tf.float32, initializer=g_init)
    # the maximum change in acceleration available given full evidence
    a_max = tf.get_variable('a_max', shape=(1,2), dtype=tf.float32, 
                            initializer = tf.random_normal_initializer())

    evidence = tf.get_variable('ev', shape=(1), dtype=tf.float32, 
                           initializer = tf.random_normal_initializer())

    delay_var = tf.get_variable('delay_var', shape=(1), dtype=tf.int32, 
                           initializer = tf.constant_initializer(30))

    mu = tf.get_variable('mu', shape=(1), dtype=tf.float32, 
                           initializer = tf.random_normal_initializer())
    sigma = tf.get_variable('sigma', shape=(1), dtype=tf.float32, 
                           initializer = tf.random_normal_initializer())

    evidence_dist = tf.contrib.distributions.Normal(mu, sigma)
    new_evidence = evidence

    X_hat, new_evidence, accumulated_evidence = model(X, a, a_max, evidence, F, G, a_prev, evidence_dist, timer_stim, delay_var, new_evidence)

    loss = tf.norm(tf.subtract(X_pred, X_hat), ord=2)
    
    # operation train minimizes the loss function
    train = tf.train.AdamOptimizer(1e-3).minimize(loss, var_list=[F, G, a_max, evidence, mu, sigma, delay_var])
    

    #optimizer = tf.train.AdamOptimizer(0.05)
    #grads_and_vars = optimizer.compute_gradients(loss, var_list=[a_max, F, G, mu, sigma])
    #train = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # intialize a saver to save trainmed model variables
    saver = tf.train.Saver()
    min_loss = 1e20
    print("\n"*4+"*"*10+"Running Graph with a session"+"*"*10+"\n"*4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_val = np.power(10.0,10.0)
        threshold = 0.1
        coords = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coords)
        count = 1
        num_records = sum(1 for _ in tf.python_io.tf_record_iterator(filenames_train))
        #while loss_val > threshold and count < 5000:
        while loss_val > threshold and count < 100:
            all_loss_values = []
            for idx in range(num_records):

                x_ord,y_ord,x_vel,y_vel,x_acc,y_acc,out_x,out_y,\
                        out_xvel,out_yvel,out_xacc,out_yacc,time_after_stim = sess.run(batch)
                
                
                _,X_hat_val,loss_val = sess.run([train,X_hat,loss],  
                                            feed_dict={X : np.array([[ x_ord[0,0],y_ord[0,0],x_vel[0,0],y_vel[0,0] ]]), 
                                            a : np.array([[ x_acc[0,0], y_acc[0,0] ]]),
                                            X_pred : np.array([[ out_x[0,0], out_y[0,0], out_xvel[0,0], out_yvel[0,0] ]]),
                                            a_prev : np.array([[ x_acc[0,0], y_acc[0,0] ]]),
                                            timer_stim : np.array([[ time_after_stim[0,0] ]]) })


                if idx%10000 == 0:
                    print("Processing record : ", idx,"\n")
                    
                if np.isnan(loss_val):
                    print(sess.run(a))
                    print(sess.run(X))
                    print(sess.run(F))
                    print("loss for {} value is {}".format(X_hat_val, loss_val))
                
                all_loss_values.append(loss_val)
                
                
            loss_val = sess.run(tf.reduce_mean(all_loss_values))
            if loss_val < min_loss:
                min_loss = loss_val
                sName = filenames_train.replace(".tfrecords",".ckpt")
                saver.save(sess,sName) # save the model as a latest_checkpoint
            print("{} : loss value is {}".format(count,loss_val))
            count = count + 1
        
        
        coords.request_stop()
        coords.join(threads)
        
    return all_loss_values

def testing(filenames_test, fname):
    ops.reset_default_graph()

    # create a string input producer to read the tfrecord
    filename_queue = tf.train.string_input_producer([filenames_test])

    # use read_and decode to retrieve tensors after casting and reshaping
    x_ord,y_ord,x_vel,y_vel,x_acc,y_acc,out_x,out_y,out_xvel,\
        out_yvel,out_xacc, out_yacc,time_after_stim = read_and_decode(filename_queue)  

    # create a batch to read input one after another
    batch = tf.train.batch([x_ord,y_ord,x_vel,y_vel,x_acc,y_acc,
                            out_x,out_y,out_xvel,out_yvel,out_xacc,out_yacc, time_after_stim], 
                           batch_size=1, capacity=200, num_threads=1)


    # variables to feed from queue
    # input vector 
    X = tf.placeholder(tf.float32, shape=[1,4], name='input_vector') 
    # control vector
    a = tf.placeholder(tf.float32, shape=[1,2], name='acceleration')
    # Prediction vector to compare loss
    X_pred = tf.placeholder(tf.float32, shape=[1, 4], name='pred_vector')
    # Previous control vector to update model state effectively
    a_prev = tf.placeholder(tf.float32, shape=[1,2], name='acceleration_prev')
    # A time based interval when stimulus comes on
    timer_stim = tf.placeholder(tf.int32, shape=[1,1], name='timer_stim')
    
    # create a session to launch the computational graph
    with tf.Session() as sess:
        # Restore model saved from training
        print(filenames_test)
        new_saver = tf.train.import_meta_graph(filenames_test.replace('tfrecords','ckpt.meta').replace('test','train'))
        new_saver.restore(sess, tf.train.latest_checkpoint('./data/'))

        # Read in model parameters from saved graph
        graph = tf.get_default_graph()
        F = graph.get_tensor_by_name("F:0")
        G = graph.get_tensor_by_name("G:0")
        a_max = graph.get_tensor_by_name("a_max:0")
        evidence = graph.get_tensor_by_name("ev:0")
        delay_var = graph.get_tensor_by_name("delay_var:0")
        mu = graph.get_tensor_by_name("mu:0")
        sigma = graph.get_tensor_by_name("sigma:0")

        # Distribution to sample from
        evidence_dist = tf.contrib.distributions.Normal(mu, sigma)
        new_evidence = evidence

        X_hat, new_evidence, accumulated_evidence = model(X, a, a_max, evidence, F, G, a_prev, evidence_dist, timer_stim, delay_var, new_evidence)
        #X_hat, accumulated_evidence = model(X, a, a_max, evidence, F, G, a_prev, evidence_dist, timer_stim, delay_var)
        
        loss = tf.norm(tf.subtract(X_pred, X_hat), ord=2)

        num_records = sum(1 for _ in tf.python_io.tf_record_iterator(filenames_test))

        print("Restored Model from checkpoint")

        coords = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coords)

        print("Stared queue Runners")

        all_loss_values = np.array([])
        all_X_hat = np.zeros([1, 4])
        all_X_val = np.zeros([1, 4])
        all_evidence = np.zeros([1,2])

        for idx in range(num_records):
            x_ord,y_ord,x_vel,y_vel,x_acc,y_acc,out_x,out_y,\
                        out_xvel,out_yvel,out_xacc,out_yacc,time_after_stim = sess.run(batch)
            X_val,X_hat_val,loss_val,evidence_val = sess.run([X_pred,X_hat,loss, accumulated_evidence],  
                                             feed_dict={X: [[ x_ord[0,0],y_ord[0,0],x_vel[0,0],y_vel[0,0] ]], 
                                             a: [[ x_acc[0,0], y_acc[0,0] ]], 
                                             X_pred: [[ out_x[0,0], out_y[0,0], out_xvel[0,0], out_yvel[0,0] ]],
                                             a_prev: [[ x_acc[0,0], y_acc[0,0] ]],
                                             timer_stim : [[ time_after_stim[0,0] ]] })
            
            if idx%10000 == 0:
                print("Processing record : ", idx,"\n")

            if np.isnan(loss_val):
                print(sess.run(a))
                print(sess.run(X))
                print(sess.run(F))
                print("loss for {} value is {}".format(X_hat_val, loss_val))

            all_loss_values = np.append( all_loss_values, np.array([loss_val]), axis=0 )
            all_X_hat = np.concatenate( (all_X_hat, X_hat_val) )
            all_X_val = np.concatenate( (all_X_val, X_val) )
            all_evidence = np.concatenate( (all_evidence, evidence_val) )

                
        all_X_hat = np.delete(all_X_hat,0,0)
        all_evidence = np.delete(all_evidence,0,0)
        all_X_val = np.delete(all_X_val,0,0)
        loss_val = sess.run(tf.reduce_mean(all_loss_values))
        print("Final loss value is {}".format(loss_val))
        np.savetxt(fname+'fitsFromModel.csv', all_X_hat, delimiter=',')
        np.savetxt(fname+'dataToFit.csv', all_X_val, delimiter=',')
        np.savetxt(fname+'evidenceFromModel.csv', all_evidence, delimiter=',')

        coords.request_stop()
        coords.join(threads)
        sess.close()

if __name__ == "__main__":
    
    # Run training and store the best results as a checkpoint #
    print("="*10+"\tTraining Model\t"+"="*10+"\n"*2)
    filenames_train = glob.glob('/home/kiran/projects/Kalman/data/S10_05_19_2017*train*subjects.tfrecords')
    for subject in filenames_train:
        print(subject)
        training(subject)
    

    # Run testing using the stored checkpoint #
    print("="*10+"\tTesting Model\t"+"="*10+"\n"*2)
    filenames_test = glob.glob('/home/kiran/projects/Kalman/data/S10_05_19_2017*test*subjects.tfrecords')
    for subject in filenames_test:
        print(subject)
        print(subject.replace('tfrecords','ckpt.meta').replace('test','train'))
        testing(subject,'')
