import numpy as np
import tensorflow as tf
#from tensorflow.python.framework import ops
import glob
import sys


"""
    NK : 13/05/2018
    need to fit an acceleration dependent on time since onset
    the dynamics are a little different for each trial, so it will be an average fit
"""

class sequenceModel:

    def __init__(self):
        #self.filenames = glob.glob('/Users/kitoo/Box Sync/Research/data_summer_2017/txtData/*.txt')
        with tf.name_scope('seqModel'):
            self.createParams()

    def createParams(self):
        # ------------- Data values -------------
        # input vector 
        self.X = tf.placeholder(tf.float64, shape=[1,4], name='input_vector') 
        # control vector
        self.a = tf.placeholder(tf.float64, shape=[1,2], name='a')
        # Prediction vector to compare loss
        self.X_pred = tf.placeholder(tf.float64, shape=[1, 4], name='pred_vector')
        # Previous control vector to update model state effectively
        self.a_prev = tf.placeholder(tf.float64, shape=[1,2], name='a_prev')
        # A time based interval when stimulus comes on
        self.timer_stim = tf.placeholder(tf.int64, shape=[], name='timer_stim')
        # Travelling from pos1 to pos2
        self.pos1 = tf.placeholder(tf.int64, shape=[], name='pos1' )
        self.pos2 = tf.placeholder(tf.int64, shape=[], name='pos2' )


        # ------------- Model Parmeters -------------
        # initialization for transition matrix F
        f_init = tf.constant([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=tf.float64)
        # transition matrix F
        self.F = tf.get_variable('F', dtype=tf.float64, initializer=f_init)

        # initialization for control vector G
        g_init = tf.constant([[1,0],[0,1],[1,0],[0,1]], dtype=tf.float64)
        # control vector G which controls accumulated evidence affecting acceleration 
        self.G = tf.get_variable('G', dtype=tf.float64, initializer=g_init)

        # the maximum change in acceleration available given full evidence
        self.a_max = tf.get_variable('a_max', shape=(1,2), dtype=tf.float64, 
                                initializer = tf.random_normal_initializer())

        # evidence accumulation rate for exponential distribution
        self.evidence = tf.get_variable('ev', shape=(1), dtype=tf.float64, 
                               initializer = tf.random_normal_initializer())

        # time on average to notice target or foil
        self.delay_var = tf.get_variable('delay_var', shape=(), dtype=tf.int64, 
                               initializer = tf.constant_initializer(30))

        self.weight_evidence = tf.get_variable('weight_evidence', shape=(1,2), dtype=tf.float64, 
                               initializer = tf.random_normal_initializer())

        # parametes for normal distribution for stochastic noise component
        self.mu = tf.get_variable('mu', shape=(1), dtype=tf.float64, 
                               initializer = tf.random_normal_initializer())
        self.sigma = tf.get_variable('sigma', shape=(1), dtype=tf.float64, 
                               initializer = tf.random_normal_initializer())

        self.zero = tf.constant([0.0,0.0], dtype=tf.float64)
        self.one = tf.constant([1.0,1.0], dtype=tf.float64)
        self.evidence_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        
        self.new_evidence_target = tf.get_variable('new_evidence_target', shape=(3,3,2), dtype=tf.float64, 
                                initializer = tf.random_normal_initializer())
        self.new_evidence_foil = tf.get_variable('new_evidence_foil', shape=(3,3,2), dtype=tf.float64, 
                                initializer = tf.random_normal_initializer())


    def cast_and_reshape(self, item, cast, shape):
        return tf.reshape(tf.cast(item, cast), shape)


    def read_and_decode(self, filename_queue):
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
            'prev_pos' : tf.FixedLenFeature([1],tf.int64),     # 21
            })
        
        # Casting serialized string to the right format
        self.subject = self.cast_and_reshape(features['Subject'], tf.int64, [1])
        self.period = self.cast_and_reshape(features['period'], tf.int64, [1])
        self.block = self.cast_and_reshape(features['block'], tf.int64, [1])
        self.stim  = self.cast_and_reshape(features['stim'], tf.int64, [1])
        self.foilInd = self.cast_and_reshape(features['foilInd'], tf.int64, [])
        self.pos  = self.cast_and_reshape(features['pos'], tf.int64, [1])
        self.trial_id  = self.cast_and_reshape(features['trial_id'], tf.int64, [1])
        self.x_ord = self.cast_and_reshape(features['x_ord'], tf.float64, [1])
        self.y_ord = self.cast_and_reshape(features['y_ord'], tf.float64, [1])
        self.x_vel = self.cast_and_reshape(features['x_vel'], tf.float64, [1])
        self.y_vel = self.cast_and_reshape(features['y_vel'], tf.float64, [1])
        self.out_x = self.cast_and_reshape(features['out_x'], tf.float64, [1])
        self.out_y = self.cast_and_reshape(features['out_y'], tf.float64, [1])
        self.out_xvel = self.cast_and_reshape(features['out_xvel'], tf.float64, [1])
        self.out_yvel = self.cast_and_reshape(features['out_yvel'], tf.float64, [1])
        self.x_acc = self.cast_and_reshape(features['x_acc'], tf.float64, [1])
        self.y_acc = self.cast_and_reshape(features['y_acc'], tf.float64, [1])
        self.out_xacc = self.cast_and_reshape(features['out_xacc'], tf.float64, [1])
        self.out_yacc = self.cast_and_reshape(features['out_yacc'], tf.float64, [1])
        self.time_after_stim  = self.cast_and_reshape(features['time_after_stim'], tf.int64, [1])
        self.prev_pos  = self.cast_and_reshape(features['prev_pos'], tf.int64, [1])     




    def model(self):
        """
        A state space model designed to predict a future state based on the current state. This is an adaptation of the kalman filter for trajectory predictions.
        
        input : X - input vector
                a - control vector
                a_max - threshold for maximum control
                F - Transition matrix for Kalman Filter
                G - Control matrix
                evidence_dist - a distribution to add noise to the process (a normal distribution here)
                time_after_stim - timing index of evenets
                new_evidence - the attractor dynamics for a target and foil

        output : X_hat - prediction for the next time point
                 accumulated_evidence - accumulated control and noise returned for later processing
                 sample_ax - rate of accumulation based on the acceleration
        """

        """
        NK : 04/16/2018 - Add two stages in process for detection and identification
        detection - process to initialize the movement
        identifiaction - process to correct the movement
        These would depend on the time for previous trial
        """

        # a_prev = tf.stop_gradient(a_prev)
        print("Running Model")
        timer_eq = tf.reshape(tf.equal(self.time_after_stim,self.delay_var), [])
        a_prev_val = tf.cond(timer_eq , 
                                lambda: tf.cond(self.foilInd, 
                                                lambda:self.new_evidence_foil[self.pos1, self.pos2,:], 
                                                lambda:self.new_evidence_target[self.pos1,self.pos2,:])
                                lambda: self.a_prev)
        # if self.time_after_stim == self.delay_var:
        #     #a_prev_val = self.a_prev
        #     a_prev_val = self.new_evidence[self.foilInd, self.pos1, self.pos2,:]
        # else:
        #     a_prev_val = self.a_prev

        stochastic_evidence = tf.add(self.evidence, self.evidence_dist.sample([1]))
        a_x = tf.maximum(self.zero, a_prev_val)
        sample_ax = tf.multiply(a_x,stochastic_evidence)
        accumulated_evidence = tf.subtract(self.one ,  tf.cast(tf.exp(tf.negative( sample_ax )), tf.float64) )
        time_delta = tf.cast( tf.subtract(self.delay_var, self.time_after_stim), tf.float64)
        a_baseline = tf.multiply(self.a, tf.multiply(tf.tanh(time_delta), self.weight_evidence) )
        a_evidence = tf.add(a_baseline , tf.multiply( self.a_max, accumulated_evidence, name='aev'))
        X_hat = tf.add(tf.matmul(self.X, self.F), tf.matmul(a_evidence, tf.transpose(self.G)), name='calculate_xhat')
        # changed new_evidence to evidence to test the variability of accumulation
        return X_hat, accumulated_evidence, sample_ax


    def getLoss(self,sess, batch, train, X_hat, loss, accumulated_evidence):
        
        x_ord, y_ord, x_vel, y_vel, x_acc, y_acc, out_x, out_y,\
        out_xvel, out_yvel, out_xacc, out_yacc, time_after_stim, block, pos, prev_pos = sess.run(batch)        

        if train:
            _,X_hat_val,loss_val, X_val, evidence_val = sess.run([train,X_hat,loss, self.X, accumulated_evidence],  
                                        feed_dict={
                                            self.X : np.array([[ x_ord[0,0], y_ord[0,0], x_vel[0,0], y_vel[0,0] ]]), 
                                            self.a : np.array([[ x_acc[0,0], y_acc[0,0] ]]),
                                            self.X_pred : np.array([[ out_x[0,0], out_y[0,0], out_xvel[0,0], out_yvel[0,0] ]]),
                                            self.a_prev : np.array([[ x_acc[0,0], y_acc[0,0] ]]),
                                            self.timer_stim : np.array( time_after_stim[0,0] ) ,
                                            self.pos1 :  np.array( prev_pos[0,0]-1 ),
                                            self.pos2 : np.array( pos[0,0]-1 ) 
                                        })
        else:
            X_hat_val,loss_val, X_val, evidence_val = sess.run([X_hat,loss, self.X, accumulated_evidence],  
                                        feed_dict={
                                            self.X : np.array([[ x_ord[0,0], y_ord[0,0], x_vel[0,0], y_vel[0,0] ]]), 
                                            self.a : np.array([[ x_acc[0,0], y_acc[0,0] ]]),
                                            self.X_pred : np.array([[ out_x[0,0], out_y[0,0], out_xvel[0,0], out_yvel[0,0] ]]),
                                            self.a_prev : np.array([[ x_acc[0,0], y_acc[0,0] ]]),
                                            self.timer_stim : np.array( time_after_stim[0,0] ) ,
                                            self.pos1 :  np.array( prev_pos[0,0]-1 ),
                                            self.pos2 : np.array( pos[0,0]-1 ) 
                                        })
        return X_hat_val, loss_val, X_val, evidence_val

    def training(self, filenames_train, fname):
        """
        function to train the model parameters using ADAM optimizer to calculate parameters minimizing prediction error

        input : filenames_train - filename of TF Record
                fname : file to save fits and parameters from the model

        output : all_loss_values
        """

        # reset the graph for each subject


        # create a string input producer to read the tfrecord
        filename_queue = tf.train.string_input_producer([filenames_train])
        # use read_and decode to retrieve tensors after casting and reshaping
        with tf.name_scope('seqModel'):
            self.read_and_decode(filename_queue)
            X_hat, accumulated_evidence, stochastic_evidence_val = self.model()
            loss = tf.norm( tf.subtract(self.X_pred, X_hat), ord=2)

            
        # operation train minimizes the loss function
        #train = tf.train.AdamOptimizer(1e-3).minimize(loss, var_list=[F, G, a_max, evidence, mu, sigma, delay_var])
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=[self.F, self.G, self.a_max, self.evidence, self.mu, self.sigma, 
                                                                self.delay_var, self.weight_evidence, self.new_evidence])
        train = optimizer.apply_gradients(grads_and_vars)

        # intialize a saver to save trainmed model variables
        saver = tf.train.Saver()
        min_loss = 1e20
        print("\n"*4+"*"*10+"Running Graph with a session"+"*"*10+"\n"*4)

        prev_block = 1

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            num_records = sum(1 for _ in tf.python_io.tf_record_iterator(filenames_train))

            with tf.name_scope('seqModel'):
                # create a batch to train the model
                batch = tf.train.batch([self.x_ord, self.y_ord, self.x_vel, self.y_vel, self.x_acc, self.y_acc,
                                self.out_x, self.out_y, self.out_xvel, self.out_yvel, self.out_xacc, self.out_yacc, 
                                self.time_after_stim, self.block, self.pos, self.prev_pos], 
                                batch_size=1, capacity=200000, num_threads=2)
                coords = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coords)
 
            
            loss_val = np.power(10.0,10.0)
            threshold = 0.1
            count = 1
            

            #while loss_val > threshold and count < 5000:
            while loss_val > threshold and count < 100:
                all_loss_values = np.array([])
                all_X_hat = np.zeros([1, 4])
                all_X_val = np.zeros([1, 4])
                all_evidence = np.zeros([1,2])
                #for idx in range(num_records):
                for idx in range(num_records):
                    with tf.name_scope('seqModel'):
                        X_hat_val, loss_val, X_val, evidence_val = self.getLoss(sess, batch, train, X_hat, loss, accumulated_evidence)
                        #evidence_val = sess.run(accumulated_evidence)
                    if idx%20000 == 0:
                        print("Processing record : ", idx,"\n")

                    if np.isnan(loss_val):
                        print(sess.run(self.X))
                        print(sess.run(self.F))
                        print("loss for {} value is {}".format(X_hat_val, loss_val))
                    #print("loss value : {} ".format(loss_val) )

                    all_loss_values = np.append( all_loss_values, np.array([loss_val]), axis=0 )
                    all_X_hat = np.concatenate( (all_X_hat, X_hat_val) )
                    all_X_val = np.concatenate( (all_X_val, X_val) )
                    all_evidence = np.concatenate( (all_evidence, evidence_val) )
                    
                    
                loss_val = sess.run(tf.reduce_mean(all_loss_values))
                if loss_val < min_loss:
                    min_loss = loss_val
                    sName = filenames_train.replace(".tfrecords",".ckpt")
                    saver.save(sess,sName) # save the model as a latest_checkpoint
                    all_X_hat = np.delete(all_X_hat,0,0)
                    all_evidence = np.delete(all_evidence,0,0)
                    all_X_val = np.delete(all_X_val,0,0)
                    loss_val = sess.run(tf.reduce_mean(all_loss_values))

                    np.savetxt(fname+'fitsFromModel_train.csv', all_X_hat, delimiter=',')
                    np.savetxt(fname+'dataToFit_train.csv', all_X_val, delimiter=',')
                    np.savetxt(fname+'evidenceFromModel_train.csv', all_evidence, delimiter=',')
                print("{} : loss value is {}".format(count,loss_val))
                count = count + 1

            

            coords.request_stop()
            coords.join(threads)
            #sess.close()
            
        return all_loss_values

    def testing(self,filenames_test, fname):
        #ops.reset_default_graph()

        # create a string input producer to read the tfrecord
        filename_queue = tf.train.string_input_producer([filenames_test])
        num_records = sum(1 for _ in tf.python_io.tf_record_iterator(filenames_test))

        # use read_and decode to retrieve tensors after casting and reshaping
        self.read_and_decode(filename_queue)

        # create a batch to read input one after another
        batch = tf.train.batch([self.x_ord, self.y_ord, self.x_vel, self.y_vel, self.x_acc, self.y_acc,
                                self.out_x, self.out_y, self.out_xvel, self.out_yvel, self.out_xacc, self.out_yacc, 
                                self.time_after_stim, self.block, self.pos, self.prev_pos], 
                               batch_size=1, capacity=40000, num_threads=1)


        # create a session to launch the computational graph
        with tf.Session() as sess:
            # Restore model saved from training
            print(filenames_test)
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            #new_saver = tf.train.import_meta_graph(filenames_test.replace('tfrecords','ckpt.meta').replace('test','train'))
            new_saver = tf.train.Saver()
            new_saver.restore(sess, filenames_test.replace('tfrecords','ckpt').replace('test','train'))

            # Read in model parameters from saved graph
            graph = tf.get_default_graph()
            with tf.name_scope('seqModel'):
                self.F = graph.get_tensor_by_name("F:0")
                self.G = graph.get_tensor_by_name("G:0")
                self.a_max = graph.get_tensor_by_name("a_max:0")
                self.evidence = graph.get_tensor_by_name("ev:0")
                self.delay_var = graph.get_tensor_by_name("delay_var:0")
                self.mu = graph.get_tensor_by_name("mu:0")
                self.sigma = graph.get_tensor_by_name("sigma:0")
                self.weight_evidence = graph.get_tensor_by_name("weight_evidence:0")
                self.new_evidence = graph.get_tensor_by_name("new_evidence:0")
                # Distribution to sample from
                self.evidence_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            
                X_hat, accumulated_evidence, stochastic_evidence_val = self.model()

                loss = tf.norm(tf.subtract(self.X_pred, X_hat), ord=2)

            print("Restored Model from checkpoint")
            coords = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coords)
            print("Started queue Runners")

            all_loss_values = np.array([])
            all_X_hat = np.zeros([1, 4])
            all_X_val = np.zeros([1, 4])
            all_evidence = np.zeros([1,2])


            for idx in range(num_records):
                with tf.name_scope('seqModel'):
                    X_hat_val, loss_val, X_val, evidence_val = self.getLoss(sess, batch, None, X_hat, loss, accumulated_evidence)
                
                if idx%20000 == 0:
                    print("Processing record : ", idx,"\n")

                if np.isnan(loss_val):
                    print(X_hat_val)
                    print(X_val)
                    print(evidence_val)
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
    filenames_train = glob.glob('/home/kiran/projects/Kalman/data/S10_05_19_2017*train*subjects.tfrecords_block*')
    ssm = sequenceModel()
    for subject in filenames_train[9:]:
        print(subject)
        ssm.training(subject, subject.replace(".tfrecords","."))


    # Run testing using the stored checkpoint #
    # print("="*10+"\tTesting Model\t"+"="*10+"\n"*2)
    # filenames_test = glob.glob('/home/kiran/projects/Kalman/data/S10_05_19_2017*test*subjects.tfrecords')
    # for subject in filenames_test:
    #     print(subject)
    #     print(subject.replace('tfrecords','ckpt.meta').replace('test','train'))
    #     ssm.testing(subject,'')
