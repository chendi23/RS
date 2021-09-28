import tensorflow as tf
#tf.enable_eager_execution()



"""
###template
def parse_example(single_example,params):
    ### expected_fatures_dict的维度由params提供
    expected_features_dict = {'Xi':}
    expected_features = tf.train.Features(
        feature=expected_features_dict
    )
    parsed_single_example = tf.io.parse_single_example(serialized=single_example, features=expected_features)

    feature1 = parsed_single_example['Xi']
    ### if features are stored in bytes
    #feature1 = tf.io.decode_raw(feature1, out_type=tf.float32)

    feature2 = parsed_single_example['Xv']
    features_dict = {'feature1': feature1, 'feature2': feature2}

    label1 = parsed_single_example['labels']
    #label2 = parsed_single_example['label2']
    labels_dict = {'label1': label1}

    return features_dict, labels_dict
"""

def parse_example(example, params):
    expected_features = {}
    expected_features['Xi'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    expected_features['Xv'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    expected_features['labels'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    parsed_feature_dict = tf.parse_single_example(example, features=expected_features)
    label = parsed_feature_dict['labels']

    label = tf.io.decode_raw(label, out_type=tf.float32)
    label = tf.reshape(label, [])
    Xi = tf.io.decode_raw(parsed_feature_dict['Xi'], out_type=tf.float32)
    Xi = tf.reshape(Xi, [10])
    Xv = tf.io.decode_raw(parsed_feature_dict['Xv'], out_type=tf.float32)
    Xv = tf.reshape(Xv, [10])
    parsed_feature_dict['Xi'] = Xi
    parsed_feature_dict['Xv'] = Xv
    parsed_feature_dict.pop('labels')

    return parsed_feature_dict, label

def input_fn(file_dir_list, params):
    files = tf.data.Dataset.list_files(file_dir_list, shuffle=False)
    """
    data_set = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10, block_length=1, sloppy=False)) \
        .map(lambda x: parse_example(x, params), num_parallel_calls=4) \
        .batch(params.batch_size) \
        .prefetch(4000)
    """
    data_set = tf.data.TFRecordDataset(files,
                                       buffer_size=params.batch_size * params.batch_size).map(lambda x: parse_example(x, params), num_parallel_calls=1).shuffle(buffer_size=params.batch_size * 10).batch(params.batch_size, drop_remainder=True).prefetch(1)

    iterator = data_set.make_one_shot_iterator()
    features_dict, labels = iterator.get_next()
    return features_dict, labels

def get_file_list(input_path):
    file_list = tf.gfile.ListDirectory(input_path)
    file_dir_list = []
    for i in file_list:
        file_dir_list.append(input_path+'\\'+i)
    print('number of train_files:', len(file_dir_list))
    return file_dir_list

