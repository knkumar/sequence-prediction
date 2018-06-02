import tensorflow as tf


def _parse_function(example_proto):
        features={
            'Subject' : tf.FixedLenFeature((), tf.int64, default_value=0),   # 1
            'period'  : tf.FixedLenFeature((), tf.int64, default_value=0),   # 2
            'block'   : tf.FixedLenFeature((), tf.int64, default_value=0),   # 3
            'stim'    : tf.FixedLenFeature((), tf.int64, default_value=0),   # 4
            'foilInd' : tf.FixedLenFeature((), tf.int64, default_value=0),   # 5
            'pos'     : tf.FixedLenFeature((), tf.int64, default_value=0),   # 6
            'trial_id': tf.FixedLenFeature((), tf.int64, default_value=0),   # 7
            'x_ord'   : tf.FixedLenFeature((), tf.float32, default_value=0),   # 8
            'y_ord'   : tf.FixedLenFeature((), tf.float32, default_value=0),   # 9
            'x_vel'   : tf.FixedLenFeature((), tf.float32, default_value=0),   # 10
            'y_vel'   : tf.FixedLenFeature((), tf.float32, default_value=0),   # 11
            'x_acc'    : tf.FixedLenFeature((), tf.float32, default_value=0),   # 12 
            'y_acc'    : tf.FixedLenFeature((), tf.float32, default_value=0),   # 13
            'out_x'   : tf.FixedLenFeature((), tf.float32, default_value=0),     # 14
            'out_y'   : tf.FixedLenFeature((), tf.float32, default_value=0),     # 15
            'out_xvel' : tf.FixedLenFeature((), tf.float32, default_value=0),   # 16
            'out_yvel' : tf.FixedLenFeature((), tf.float32, default_value=0),   # 17
            'out_xacc' : tf.FixedLenFeature((), tf.float32, default_value=0),   # 18
            'out_yacc' : tf.FixedLenFeature((), tf.float32, default_value=0),   # 19
            'time_after_stim' : tf.FixedLenFeature((),tf.int64, default_value=0),     # 20
            'prev_pos' : tf.FixedLenFeature((),tf.int64, default_value=0),          # 21
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

        # This is an op that gets the next element from the iterator
        next_element = iterator.get_next()

        # initialize the iterator to get data
        dataset_init_op = iterator.make_initializer(dataset)

        # return both the iterator generator and the initialization op
        return next_element, dataset_init_op
