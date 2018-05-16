import numpy as np
import tensorflow as tf
import pandas as pd
import glob
filenames = glob.glob('/home/kiran/projects/hmm-mouse/data/txtData/S10_05_19_2017*.txt')


def _bytes_feature(value):
    """ 
    Function to convert value to a bytesList feature
    input : value
    output : ByteList Feature object
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """ 
    Function to convert value to a int64List feature
    input : value
    output : int64List Feature object
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """ 
    Function to convert value to a floatList feature
    input : value
    output : floatList Feature object
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def processText(data, pos):
    """
    Function to process Text data and convert labels to numeric values
    input : data - data to be processed
            pos - position indicator to return positions of events and a foil indicator of events
    output : dataRet - data with trial number for each event
             posRet - position of events in the sequence
             foilRet - foil Indicator for each event
    """
    # dictionary to keep track of data processed
    dataDict = {}
    # count variable to monitor events
    count = 0
    # list to return data with
    dataRet = []
    
    # create lists only if needed to return
    if pos:
        # list to return position
        posRet = []
        # list to return foils
        foilRet = []
        # consistent labels for different events
        foils = {'R':1,'S':2,'Q':3}
    
    prev_item = ''

    for item in data:
        # if postition is required
        if pos:
            # check if item is foil
            checkFoil = chr(item[1])
            if checkFoil in foils:
                # store item type in data to return
                dataRet.append(foils[chr(item[1])])
                # store item position
                posRet.append(int(chr(item[2])))
                # store flag for event type
                foilRet.append(1)
            else:
                # store item type in data to return
                dataRet.append(foils[chr(item[0])])
                # store item position
                posRet.append(int(chr(item[1])))
                # store flag for event type
                foilRet.append(0)
        else:
            # if a new event occurs
            if prev_item != item:
                # increment indicator variable
                count = count +1
                # append indicator variable to data for returning
                dataRet.append(count)
            else:
                # append indicator variable to data for returning
                dataRet.append(count)
            prev_item = item
    if pos:
        return(dataRet, foilRet, posRet)
    else:
        return(dataRet)



def writeTFrecords(tfrecords_filename, filenames, prediction_time):
    """
    Function to write TF records processing text files
    input : tfrecords_filename - filename to write TF records in
            filenames - filenames to process
    """
    # intialize a file identifier
    subjectId = 0
    # process all filenames into a training and testing data -TF records
    for file in filenames:
        # numpy loadtxt for file with column names and formats
        data_cond = np.loadtxt(file,dtype={'names': ['Period', 'Block', 'Trial','Trial_id','x_ord','y_ord'],  
                    'formats': ['S3', 'S7' ,'S6','i4', 'i4', 'i4']}, delimiter="\t",skiprows=1)
        # name to save TF records
        sName = file.replace('.txt','')
        saveName = sName.split("/")
        # display current file being processed
        tfrecords_train_savename = "data/"+saveName[-1]+"_train_"+tfrecords_filename
        print(tfrecords_train_savename)
        tfrecords_test_savename = "data/"+saveName[-1]+"_test_"+tfrecords_filename
        # open recordwriters for training and testing data
        
        testWriter = tf.python_io.TFRecordWriter(tfrecords_test_savename)
        
        # process text to convert text labels to numerical indicators
        period = processText(data_cond['Period'],0)
        block  = processText(data_cond['Block'],0)
        [stim, foil, pos]  = processText(data_cond['Trial'],1) 
        # read input data
        x_ord = data_cond['x_ord']
        y_ord = data_cond['y_ord']
        trial_id = data_cond['Trial_id']
        
        # process input data to create dervied vectors
        x_diff = np.append(0.0,np.diff(x_ord))
        y_diff = np.append(0.0,np.diff(y_ord))
        thetas = np.arctan2(y_diff, x_diff)
        speed = np.sqrt((x_diff*x_diff) + (y_diff*y_diff))
        x_vel = speed * np.cos(thetas)
        y_vel = speed * np.sin(thetas)
        x_acc = np.append(0.0, np.diff(x_vel))
        y_acc = np.append(0.0, np.diff(y_vel))
        
        # store data from future in the same example to feed into algorithm
        out_x = np.append(x_ord[prediction_time:],[-1]*prediction_time)
        out_y = np.append(y_ord[prediction_time:],[-1]*prediction_time)

        out_xacc = np.append([0.0]*prediction_time, x_acc[0:(len(x_acc)-prediction_time)] )
        out_yacc = np.append([0.0]*prediction_time, y_acc[0:(len(y_acc)-prediction_time)] )

        out_xvel = np.append(x_vel[prediction_time:], [-1]*prediction_time)
        out_yvel = np.append(y_vel[prediction_time:], [-1]*prediction_time)
    
        subjectId = subjectId + 1
        trial_id_prev = 0
        timer = 0
    
        # generate an example for each time point
        prev_block = 0
        for idx in range(len(period)):
            # schedule new information for events
            if prev_block != block[idx]:
                print(period[idx],block[idx])
                trainWriter = tf.python_io.TFRecordWriter(
                    tfrecords_train_savename+"_block"+str(period[idx])+str(block[idx]))

            if trial_id_prev != trial_id[idx]:
                timer = 1
                trial_id_prev = trial_id[idx]
                prev_pos = pos[idx]
                
            
            # generate example with features
            example = tf.train.Example(features=tf.train.Features(feature={
                'Subject' : _int64_feature(subjectId),      # 1
                'period'  : _int64_feature(period[idx]),    # 2
                'block'   : _int64_feature(block[idx]),     # 3
                'stim'    : _int64_feature(stim[idx]),      # 4
                'foilInd' : _int64_feature(foil[idx]),      # 5
                'pos'     : _int64_feature(pos[idx]),       # 6
                'trial_id': _int64_feature(trial_id[idx]),  # 7
                'x_ord'   : _int64_feature(x_ord[idx]),     # 8
                'y_ord'   : _int64_feature(y_ord[idx]),     # 9
                'x_vel'   : _float_feature(x_vel[idx]),     # 10
                'y_vel'   : _float_feature(y_vel[idx]),     # 11
                'x_acc'   :  _float_feature(x_acc[idx]),    # 12
                'y_acc'   :  _float_feature(y_acc[idx]),    # 13
                'out_x'   : _int64_feature(out_x[idx]),     # 14
                'out_y'   : _int64_feature(out_y[idx]),     # 15
                'out_xvel' : _float_feature(out_xvel[idx]), # 16
                'out_yvel' : _float_feature(out_yvel[idx]), # 17
                'out_xacc' : _float_feature(out_xacc[idx]), # 18
                'out_yacc' : _float_feature(out_yacc[idx]),  # 19
                'time_after_stim' : _int64_feature(timer),   # 20
                'prev_pos' : _int64_feature(prev_pos)
            }))
            
            timer = timer+1
            prev_block = block[idx]
            trainWriter.write(example.SerializeToString())
            testWriter.write(example.SerializeToString())
    
        trainWriter.close()
        testWriter.close()
        


tfrecords_filename = 'mouse_subjects.tfrecords'

writeTFrecords(tfrecords_filename,filenames, 1)
print("Finished generating TFRecords for training and testing")

