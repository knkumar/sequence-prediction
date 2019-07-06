import tensorflow as tf


# def _parse_function(example_proto):
#         features={
#             'Subject' : tf.VarLenFeature(tf.int64),   # 1
#             'period'  : tf.VarLenFeature(tf.int64),   # 2
#             'block'   : tf.VarLenFeature(tf.int64),   # 3
#             'stim'    : tf.VarLenFeature(tf.int64),   # 4
#             'foilInd' : tf.VarLenFeature(tf.int64),   # 5
#             'pos'     : tf.VarLenFeature(tf.int64),   # 6
#             'trial_id': tf.VarLenFeature(tf.int64),   # 7
#             'x_ord'   : tf.VarLenFeature(tf.float32),   # 8
#             'y_ord'   : tf.VarLenFeature(tf.float32),   # 9
#             'x_vel'   : tf.VarLenFeature(tf.float32),   # 10
#             'y_vel'   : tf.VarLenFeature(tf.float32),   # 11
#             'x_acc'    : tf.VarLenFeature(tf.float32),   # 12 
#             'y_acc'    : tf.VarLenFeature(tf.float32),   # 13
#             'out_x'   : tf.VarLenFeature(tf.float32),     # 14
#             'out_y'   : tf.VarLenFeature(tf.float32),     # 15
#             'out_xvel' : tf.VarLenFeature(tf.float32),   # 16
#             'out_yvel' : tf.VarLenFeature(tf.float32),   # 17
#             'out_xacc' : tf.VarLenFeature(tf.float32),   # 18
#             'out_yacc' : tf.VarLenFeature(tf.float32),   # 19
#             'time_after_stim' : tf.VarLenFeature(tf.int64),     # 20
#             'prev_pos' : tf.VarLenFeature(tf.int64),          # 21
#             }
#         parsed_features = tf.parse_single_example(example_proto, features)
#         return parsed_features

def _parse_function(example_proto):
        features={
            'Subject' : tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True, default_value=0),   # 1
            'period'  : tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True, default_value=0),   # 2
            'block'   : tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True, default_value=0),   # 3
            'stim'    : tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True, default_value=0),   # 4
            'foilInd' : tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True, default_value=0),   # 5
            'pos'     : tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True, default_value=0),   # 6
            'trial_id': tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True, default_value=0),   # 7
            'x_ord'   : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 8
            'y_ord'   : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 9
            'x_vel'   : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 10
            'y_vel'   : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 11
            'x_acc'    : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 12 
            'y_acc'    : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 13
            'out_x'   : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),     # 14
            'out_y'   : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),     # 15
            'out_xvel' : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 16
            'out_yvel' : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 17
            'out_xacc' : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 18
            'out_yacc' : tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True, default_value=0),   # 19
            'time_after_stim' : tf.FixedLenSequenceFeature((),tf.int64, allow_missing=True, default_value=0),     # 20
            'prev_pos' : tf.FixedLenSequenceFeature((),tf.int64, allow_missing=True, default_value=0),          # 21
            }
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features

def create_dataset( filenames):
        """
        Function to read and decode TF records for further processsing - 
            Read serialized file TF record and cast features variables to the right format
            This has to be changed for a new input pipeline.

        input  : filenames - the string or list for ingesting TF records
        output : variables to feed 

        """

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function, num_parallel_calls=1) # Parse the record into tensors.
        # create a reinitializable iterator. 
        # This enables to set up an initializer for different input data
        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
        # initialize the iterator to get data
        dataset_init_op = iterator.make_initializer(dataset)

        # This is an op that gets the next element from the iterator
        next_element = iterator.get_next()

        
        # return both the iterator generator and the initialization op
        return next_element, dataset_init_op
