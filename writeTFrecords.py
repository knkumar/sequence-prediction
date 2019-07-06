import numpy as np
import tensorflow as tf
import glob

filenames = glob.glob('/home/kiran/projects/hmm-mouse/data/txtData/S*.txt')


def _bytes_feature(value):
    """ 
    Function to convert value to a bytesList feature
    input : value
    output : ByteList Feature object
    """
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
    """ 
    Function to convert value to a int64List feature
    input : value
    output : int64List Feature object
    """
    # return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """ 
    Function to convert value to a floatList feature
    input : value
    output : floatList Feature object
    """
    # return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def processText(data, pos, period = None):
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

    for idx,item in enumerate(data):
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
            if prev_item != item or (np.size(period) > 1 and period[idx-1] != period[idx]):
                # increment indicator variable
                count = count +1
                # append indicator variable to data for returning
                dataRet.append(count)
            else:
                # append indicator variable to data for returning
                dataRet.append(count)
            prev_item = item
    if pos:
        return(np.array(dataRet), np.array(foilRet), np.array(posRet))
    else:
        return(np.array(dataRet))



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
        print(file)
        data_cond = np.loadtxt(file,dtype={'names': ['Period', 'Block', 'Trial','Trial_id','x_ord','y_ord'],  
                    'formats': ['S3', 'S7' ,'S6','i4', 'i4', 'i4']}, delimiter="\t",skiprows=1)
        # name to save TF records
        sName = file.replace('.txt','')
        saveName = sName.split("/")
        # display current file being processed
        tfrecords_train_savename = "data/tfrecords/"+saveName[-1]+"_train_"+tfrecords_filename
        print(tfrecords_train_savename)
        tfrecords_test_savename = "data/tfrecords/"+saveName[-1]+"_test_"+tfrecords_filename
        # open recordwriters for training and testing data
        testWriter = tf.io.TFRecordWriter(tfrecords_test_savename+'.tfrecords')
        
        # process text to convert text labels to numerical indicators
        period = processText(data_cond['Period'],0)
        print(period.shape)
        block  = processText(data_cond['Block'],0, period)
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
        time_after_stim = np.array([],dtype=np.int32)
        prev_pos_arr = np.array([],dtype=np.int32)
        uniq_block = np.unique(block)
        prev_pos = 1

        for idx,trial_num in enumerate(trial_id):
            if trial_id_prev != trial_id[idx]:
                timer = 1
                trial_id_prev = trial_id[idx]
                if idx > 0:
                    prev_pos = pos[idx-1]
            time_after_stim = np.append(time_after_stim,timer)
            prev_pos_arr = np.append(prev_pos_arr,prev_pos)
            timer = timer+1

        for curr_block in uniq_block:
            # open recordwriters for training and testing data
            blk_ids = np.where(block == curr_block)[0]  
            trainWriter = tf.io.TFRecordWriter(tfrecords_train_savename+'_block_'+str(curr_block)+'.tfrecords')
            # print(np.shape(blk_ids), type(blk_ids))
            # generate example with features
            example = tf.train.Example(features=tf.train.Features(feature={
                'Subject' : _int64_feature(np.repeat(subjectId,np.size(blk_ids)) ),      # 1
                'period'  : _int64_feature(period[blk_ids]),    # 2
                'block'   : _int64_feature(block[blk_ids]),     # 3
                'stim'    : _int64_feature(stim[blk_ids]),      # 4
                'foilInd' : _int64_feature(foil[blk_ids]),      # 5
                'pos'     : _int64_feature(pos[blk_ids]),       # 6
                'trial_id': _int64_feature(trial_id[blk_ids]),  # 7
                'x_ord'   : _float_feature(x_ord[blk_ids]),     # 8
                'y_ord'   : _float_feature(y_ord[blk_ids]),     # 9
                'x_vel'   : _float_feature(x_vel[blk_ids]),     # 10
                'y_vel'   : _float_feature(y_vel[blk_ids]),     # 11
                'x_acc'   :  _float_feature(x_acc[blk_ids]),    # 12
                'y_acc'   :  _float_feature(y_acc[blk_ids]),    # 13
                'out_x'   : _float_feature(out_x[blk_ids]),     # 14
                'out_y'   : _float_feature(out_y[blk_ids]),     # 15
                'out_xvel' : _float_feature(out_xvel[blk_ids]), # 16
                'out_yvel' : _float_feature(out_yvel[blk_ids]), # 17
                'out_xacc' : _float_feature(out_xacc[blk_ids]), # 18
                'out_yacc' : _float_feature(out_yacc[blk_ids]),  # 19
                'time_after_stim' : _int64_feature(time_after_stim[blk_ids]),   # 20
                'prev_pos' : _int64_feature(prev_pos_arr[blk_ids])        # 21
            }))

            trainWriter.write(example.SerializeToString())
            testWriter.write(example.SerializeToString())
            trainWriter.close()

        testWriter.close()
        
tfrecords_filename = 'mouse'

writeTFrecords(tfrecords_filename,filenames, 1)
print("Finished generating TFRecords for training and testing")

