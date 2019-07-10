import glob
import sys
import os
import pathlib
from sequence_model import sequenceModel

"""
    NK : 13/05/2018
    need to fit an acceleration dependent on time since onset
    the dynamics are a little different for each trial, so it will be an average fit
"""


if __name__ == "__main__":
    # Run training and store the best results as a checkpoint #
    print("="*10+"\tTraining Model\t"+"="*10+"\n"*2)
    filenames_train = glob.glob('/home/kiran/projects/Kalman/data/tfrecords/S1*train*.tfrecords')
    # filenames_train = glob.glob('/scratch/knkumar/Kalman/data/tfrecords/S1_*train*.tfrecords')
    done = glob.glob('/scratch/knkumar/Kalman/data/S1*/*fits*')
    done = [item.split('/')[-1].replace('fitsFromModel_train.csv','tfrecords') for item in done]

    for subject in filenames_train:
        if subject.split('/')[-1] in done:
            continue
        ssm = None
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
