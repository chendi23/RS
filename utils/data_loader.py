import tensorflow as tf


def parse_example(single_example):
    expected_features_dict = {}
    expected_features = tf.train.Features(
        feature=expected_features_dict
    )
    parsed_single_example = tf.io.parse_single_example(serialized=single_example, features=expected_features)

    feature1 = parsed_single_example['feature1']
    feature2 = parsed_single_example['feature2']
    features_dict = {'feature1': feature1, 'feature2': feature2}

    label1 = parsed_single_example['label1']
    label2 = parsed_single_example['label2']
    labels_dict = {'label1': label1, 'label2': label2}

    return features_dict, labels_dict


def input_fn(file_dir_list, params):
    files = tf.data.Dataset.list_files(file_dir_list, shuffle=False)
    data_set = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10, block_length=1, sloppy=False)) \
        .map(lambda x: parse_example(x, params), num_parallel_calls=4) \
        .batch(params.batch_size) \
        .prefetch(4000)

    iterator = data_set.make_one_shot_iterator()
    features_dict, labels_dict = iterator.get_next()
    return features_dict, labels_dict
