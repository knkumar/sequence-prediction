import numpy as np
import tensorflow as tf
#from tensorflow.python.framework import ops
import glob
import sys
import os
import pathlib
from create_dataset import create_dataset

"""
    NK : 13/05/2018
    need to fit an acceleration dependent on time since onset
    the dynamics are a little different for each trial, so it will be an average fit
"""

class sequenceModel:

    def __init__(self, filenames):
        """
        constructor to create a sequenceModel object 
        identify the number of records in the dataset
        """
        #self.filenames = glob.glob('/Users/kitoo/Box Sync/Research/data_summer_2017/txtData/*.txt')
        tf.reset_default_graph()
        with tf.name_scope('seqModel'):
            # get number of records in the dataset
            self.num_records = sum(1 for _ in tf.python_io.tf_record_iterator(filenames))
            # create a dataset iterator to pass to the model
            next_element, dataset_init_op = create_dataset(filenames)
            # use iterator generator to create data variables
            self.createDataIterators(next_element)
            self.createParams()
            self.dataset_init_op = dataset_init_op
        self.sess = tf.Session()

    def __enter__(self):
        return self

    # perform cleanup and close the opened session
    def __exit__(self,exc_type, exc_value, traceback):
        if exc_type is not None:
            print(exc_type, exc_value, traceback)
        self.sess.close()


    def createDataIterators(self, inputs) :
        """
        function uses the dataset iterator object to create data variables for the model

        input : inputs - dataset iterator object

        output: None

        """
        # ------------- Data values -------------
        # input vector
        self.X = tf.cast(tf.stack([[ inputs['x_ord'],inputs['y_ord'], inputs['x_vel'], inputs['y_vel'] ]],axis=0), tf.float64)
        # control vector
        self.a = tf.cast(tf.stack([[ inputs['x_acc'], inputs['y_acc'] ]], axis=0), tf.float64)
        # Prediction vector to compare loss
        self.X_pred = tf.cast(tf.stack([[ inputs['out_x'], inputs['out_y'], inputs['out_xvel'], inputs['out_yvel'] ]], axis=0), tf.float64)
        # Previous control vector to update model state effectively
        self.a_prev = tf.cast(tf.stack([[ inputs['out_xacc'], inputs['out_yacc'] ]], axis=0), tf.float64)
        # A time based interval when stimulus comes on
        self.time_after_stim = inputs['time_after_stim']
        # Travelling from pos1 to pos2
        self.pos1 = inputs['prev_pos']-1
        self.pos2 = inputs['pos']-1
        self.foilInd = inputs['foilInd']
        

    def createParams(self):
        """
        function to create model parameters using tensorflow variables

        input : None

        output: None

        """
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
        # weight evidence based on time since detection
        self.weight_evidence = tf.get_variable('weight_evidence', shape=(1,2), dtype=tf.float64, 
                               initializer = tf.random_normal_initializer())
        # parametes for normal distribution for stochastic noise component
        self.mu = tf.get_variable('mu', shape=(1), dtype=tf.float64, 
                               initializer = tf.random_normal_initializer())
        self.sigma = tf.get_variable('sigma', shape=(1), dtype=tf.float64, 
                               initializer = tf.random_normal_initializer())
        # a zero and one tensor for graph computations
        self.zero = tf.constant([0.0,0.0], dtype=tf.float64)
        self.one = tf.constant([1.0,1.0], dtype=tf.float64)
        # distribution to sample evidence from
        self.evidence_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        
        # attractor dynamics for target and foil movements 
        # from start position to destination position
        self.attractor_dynamics_target = tf.get_variable('attractor_dynamics_target', shape=(3,3,2), dtype=tf.float64, 
                                initializer = tf.random_normal_initializer())
        self.attractor_dynamics_foil = tf.get_variable('attractor_dynamics_foil', shape=(3,3,2), dtype=tf.float64, 
                                initializer = tf.random_normal_initializer())

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
                attractor_dynamics - the attractor dynamics for a target and foil

        output : X_hat - prediction for the next time point
                 accumulated_evidence - accumulated control and noise returned for later processing
                 sample_ax - rate of accumulation based on the acceleration
        """

        """
        NK : 04/16/2018 - Add two stages in process for detection and identification
        detection - process to initialize the movement
        identification - process to correct the movement
        These would depend on the time for previous trial
        """

        # a_prev = tf.stop_gradient(a_prev)
        print("Running Model")
        timer_eq = tf.reshape(tf.equal(self.time_after_stim,self.delay_var), [])
        foil_eq = tf.reshape(tf.equal(self.foilInd,np.array([1],dtype='int64')), [])
        a_prev_val = tf.cond(timer_eq , 
                            lambda: tf.cond(foil_eq,
                                    lambda:self.attractor_dynamics_foil[self.pos1, self.pos2,:], 
                                    lambda:self.attractor_dynamics_target[self.pos1, self.pos2,:]), 
                            lambda: self.a_prev)

        stochastic_evidence = tf.add(self.evidence, self.evidence_dist.sample([1]))
        a_x = tf.maximum(self.zero, a_prev_val)
        sample_ax = tf.multiply(a_x, stochastic_evidence)
        accumulated_evidence = tf.subtract(self.one ,  tf.cast(tf.exp(tf.negative( sample_ax )), tf.float64) )
        time_delta = tf.cast( tf.subtract(self.delay_var, self.time_after_stim), tf.float64)
        a_baseline = tf.multiply(self.a, tf.multiply(tf.tanh(time_delta), self.weight_evidence) )
        a_evidence = tf.add(a_baseline , tf.multiply( self.a_max, accumulated_evidence, name='aev'))
        X_hat = tf.add(tf.matmul(self.X, self.F), tf.matmul(a_evidence, tf.transpose(self.G)), name='calculate_xhat')
        # changed attractor_dynamics to evidence to test the variability of accumulation
        return X_hat, accumulated_evidence, sample_ax


    def getLoss(self,sess, train, X_hat, loss, accumulated_evidence):
        if train:
            _,X_hat_val,loss_val, X_val, evidence_val = sess.run([train,X_hat,loss, self.X, accumulated_evidence])
        else:
            X_hat_val,loss_val, X_val, evidence_val = sess.run([X_hat,loss, self.X, accumulated_evidence])
        return X_hat_val, loss_val, X_val, evidence_val

    def training(self, filenames_train, fname):
        """
        function to train the model parameters using ADAM optimizer to calculate parameters minimizing prediction error

        input : filenames_train - filename of TF Record

        output : all_loss_values
        """        
        
        # use read_and decode to retrieve tensors after casting and reshaping
        with tf.name_scope('seqModel'):
            X_hat, accumulated_evidence, stochastic_evidence_val = self.model()
            loss = tf.norm( tf.subtract(self.X_pred, X_hat), ord=2)
            
        # operation train minimizes the loss function
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=[self.F, self.G, self.a_max, self.evidence, self.mu, self.sigma, 
                                                                self.delay_var, self.weight_evidence, 
                                                                self.attractor_dynamics_foil, self.attractor_dynamics_target])
        train = optimizer.apply_gradients(grads_and_vars)

        # intialize a saver to save trained model variables
        saver = tf.train.Saver()

        min_loss = 1e20
        print("\n"*4+"*"*10+"Running Graph with a session"+"*"*10+"\n"*4)

        # run initializers for local and global variables in the model
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        
        loss_val = np.power(10.0,10.0)
        threshold = 0.1
        count = 1

        for _ in range(150):
            self.sess.run(self.dataset_init_op)
            all_loss_values = np.array([])
            all_X_hat = np.zeros([1, 4])
            all_X_val = np.zeros([1, 4])
            all_evidence = np.zeros([1,2])
            idx = 0
            while True:
                try:
                    with tf.name_scope('seqModel'):
                        X_hat_val, loss_val, X_val, evidence_val = self.getLoss(self.sess, train, X_hat, loss, accumulated_evidence)
                        idx += 1
                    if idx%10000 == 0:
                        print("Processing record : ", idx,"\n")
                    if np.isnan(loss_val):
                        self.sess.run(tf.local_variables_initializer())
                        self.sess.run(tf.global_variables_initializer())
                        print("loss for {} value is {}".format(X_hat_val, loss_val))
                    all_loss_values = np.append( all_loss_values, np.array([loss_val]), axis=0 )
                    all_X_hat = np.concatenate( (all_X_hat, X_hat_val) )
                    all_X_val = np.concatenate( (all_X_val, X_val) )
                    all_evidence = np.concatenate( (all_evidence, evidence_val) )

                except tf.errors.OutOfRangeError:
                    break

            loss_val = self.sess.run(tf.reduce_mean(all_loss_values))
            if loss_val < min_loss:
                min_loss = loss_val
                sName = filenames_train.replace(".tfrecords",".ckpt")
                saver.save(self.sess,sName) # save the model as a latest_checkpoint
                all_X_hat = np.delete(all_X_hat,0,0)
                all_evidence = np.delete(all_evidence,0,0)
                all_X_val = np.delete(all_X_val,0,0)
                loss_val = self.sess.run(tf.reduce_mean(all_loss_values))
                # if the parameter file already exists remove it
                # create a new parameter file to append varilables in readable format
                if os.path.isfile(fname+"_parameters_from_fit.csv"):
                    os.remove(fname+"_parameters_from_fit.csv")
                # open a file handle to save parameters
                f = open(fname+"_parameters_from_fit.csv","ab")
                np.savetxt(f, min_loss, header="Loss Value \n", delimiter='\t', fmt="%f")
                np.savetxt(f,self.sess.run(self.F), header="Parameter- F\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.G), header="Parameter- G\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.a_max), header="Parameter- a_max\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.evidence), header="Parameter- evidence\n", delimiter='\t')
                np.savetxt(f,[self.sess.run(self.delay_var)], header="Parameter- delay_var\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.weight_evidence), header="Parameter- weight_evidence\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.mu), header="Parameter- mu\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.sigma), header="Parameter- sigma\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.attractor_dynamics_target[0,:,:]), header="Parameter- attractor_dynamics 0_1\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.attractor_dynamics_target[1,:,:]), header="Parameter- attractor_dynamics 0_2\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.attractor_dynamics_target[2,:,:]), header="Parameter- attractor_dynamics 0_3\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.attractor_dynamics_foil[0,:,:]), header="Parameter- attractor_dynamics 1_1\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.attractor_dynamics_foil[1,:,:]), header="Parameter- attractor_dynamics 1_2\n", delimiter='\t')
                np.savetxt(f,self.sess.run(self.attractor_dynamics_foil[2,:,:]), header="Parameter- attractor_dynamics 1_3\n", delimiter='\t')
                f.close()

                np.savetxt(fname+'fitsFromModel_train.csv', all_X_hat, header="fits From Model\n",delimiter=',')
                np.savetxt(fname+'dataToFit_train.csv', all_X_val, header="data to fit\n", delimiter=',')
                np.savetxt(fname+'evidenceFromModel_train.csv', all_evidence, header="evidence From Model\n", delimiter=',')
            print("{} : loss value is {}".format(count,loss_val))
            count = count + 1         
        
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
            new_saver = tf.train.import_meta_graph(filenames_test.replace('tfrecords','ckpt.meta').replace('test','train'))
            new_saver.restore(sess, tf.train.latest_checkpoint('./data/'))

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
                self.attractor_dynamics = graph.get_tensor_by_name("attractor_dynamics:0")
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
    filenames_train = glob.glob('/scratch/knkumar/Kalman/data/tfrecords/S*train*subjects.tfrecords_block*')
    for subject in filenames_train:
    	with sequenceModel(subject) as ssm:
            fname = subject.replace(".tfrecords",".").replace("tfrecords/","")
            #print(fname)
            sname = fname.split("train")[0]
            sname = sname.split("/")[-1]
            fname = fname.replace("data/","data/"+sname[:-1]+"/")
            fname = fname.replace("train_mouse_subjects._","") 
            dfname = ("/").join(fname.split("/")[:-1])
            pathlib.Path(dfname).mkdir(parents=True, exist_ok=True)
            ssm.training(subject, fname)


    # Run testing using the stored checkpoint #
    # print("="*10+"\tTesting Model\t"+"="*10+"\n"*2)
    # filenames_test = glob.glob('/home/kiran/projects/Kalman/data/S10_05_19_2017*test*subjects.tfrecords')
    # for subject in filenames_test:
    #     print(subject)
    #     print(subject.replace('tfrecords','ckpt.meta').replace('test','train'))
    #     ssm.testing(subject,'')
