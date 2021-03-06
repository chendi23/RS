import numpy as np
import os
from datetime import datetime
import tensorflow as tf
import pandas as pd
from copy import deepcopy
tf.enable_eager_execution()

def pre_processing(path):
    return


class FeatureDictionary:
    def __init__(self, df=None, numeric_cols=['age'], ignore_cols=['click', 'heat', 'grade_id','grade_name','u_class_label','i_entities_label','user_name', 'id','u_entities_label','i_keywords_label', 'u_keywords_label']):
        assert not(df is None)
        self.df = df
        self.rows_count = self.df.shape[0]
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.field_size = 0


    def gen_feature_dictionary(self):
        feature_dict = {}
        col_count = 0
        for col in self.df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                feature_dict[col] = col_count
                col_count += 1
                self.field_size += 1
            else:
                us = self.df[col].unique()
                feature_dict[col] = dict(zip(us, range(col_count, col_count+len(us))))
                col_count += len(us)
                self.field_size += 1
        self.feature_dim = col_count
        #print(feature_dict)

        return feature_dict




class DataParser:
    def __init__(self, feature_dict_ob):
        assert isinstance(feature_dict_ob, FeatureDictionary)
        self.feature_dict_ob = feature_dict_ob
        self.feature_dict = self.feature_dict_ob.gen_feature_dictionary()
        self.feature_dim = self.feature_dict_ob.feature_dim
        self.field_size = self.feature_dict_ob.field_size
        self.rows_count= self.feature_dict_ob.rows_count

    def parse(self, df=None):
        assert not(df is None)
        labels = df['click'].tolist()
        dfi = deepcopy(df)
        dfv = deepcopy(df)
        for i in dfi.columns:
            if i in self.feature_dict_ob.ignore_cols:
                dfi.drop([i], axis=1, inplace=True)
                dfv.drop([i], axis=1, inplace=True)
            elif i in self.feature_dict_ob.numeric_cols:
                dfi[i] = self.feature_dict[i]
            else:
                dfi[i] = dfi[i].map(self.feature_dict[i])
                dfv[i] = 1

        Xi = dfi.values.tolist()
        Xv = dfv.values.tolist()

        return Xi, Xv, labels





def get_ByteFeature(value):
    value = value.encode('utf-8')
    value = [value]
    byte_list = tf.train.BytesList(value=value)
    return tf.train.Feature(byte_list)

def get_Float_ListFeature(value):
    if not isinstance(value, np.ndarray):
        value = np.asarray(value)
        value = value.astype(np.float32).tostring()
        value = [value]
        float_list = tf.train.BytesList(value=value)
        return tf.train.Feature(bytes_list=float_list)
    else:
        value = value.astype(np.float32).tostring()
        value = [value]
        float_list = tf.train.BytesList(value=value)
        return tf.train.Feature(float_list)


def get_Int_ListFeature(value):
    value = np.asarray(value)
    value= value.astype(np.int64).tostring()
    value = [value]
    int_list = tf.train.Int64List(value=value)
    return tf.train.Feature(int_list)

def get_LabelFeature(value):
    value = [value]
    float_list = tf.train.FloatList(value=value)
    return tf.train.Feature(float_list)



def list_to_tfrecords(lists_dict=None):
    assert not (lists_dict is None)
    output_dir = 'tf_record_from_lists'
    if not os.path.exists(os.path.join(output_dir)):
        os.mkdir(output_dir)
    filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.tfrecords'
    with tf.io.TFRecordWriter(path=os.path.join(output_dir, filename)) as wr:
        for i in range(rows_count):
            single_row_dict = {}
            for k, v in lists_dict.items():
                single_row_dict[k] = get_Float_ListFeature(v[i])
                #print(single_row_dict)
            features = tf.train.Features(feature=single_row_dict)
            exanple = tf.train.Example(features=features)
            #print(exanple)
            wr.write(record=exanple.SerializeToString())

        wr.close()

    return




def parse_example(example):
    expected_features = {}
    expected_features['Xi'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    expected_features['Xv'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    expected_features['labels'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    parsed_feature_dict = tf.io.parse_single_example(example, features=expected_features)
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


def parse_tfrecords(tfrecords_path):
    with tf.Session() as sess:
        dataset = tf.data.TFRecordDataset([tfrecords_path])
        dataset = dataset.map(lambda x:parse_example(x))
        print(tf.data.get_output_shapes(dataset))

    return


'''test'''

"""
df_input = pd.read_csv('D:\\zcd\\processed_csv.csv')
df_input = df_input.drop(columns=['Unnamed: 0'])
fd_object = FeatureDictionary(df=df_input)
ps = DataParser(fd_object)
feature_dim, rows_count, field_size = ps.feature_dim, ps.rows_count, ps.field_size
print('feature_dim: ', feature_dim, 'row_count: ', rows_count, 'field_size: ', field_size)
Xi, Xv, labels = ps.parse(df_input)

lists_dict = {'Xi': Xi, 'Xv': Xv, 'labels': labels}
#list_to_tfrecords(lists_dict)

tf_records_path = 'D:\\zcd\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\utils\\tf_record_from_lists'
"""

"""
path = './tf_record_from_lists/2021_09_27_09_28_10.tfrecords'
dataset = tf.data.TFRecordDataset([path])

expected_features = {}
expected_features['Xi'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
expected_features['Xv'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
expected_features['labels'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

dataset = dataset.map(parse_example)
i = tf.compat.v1.data.make_one_shot_iterator(dataset)
_, t = i.get_next()
with tf.Session() as sess:
    for j in range(10):
        print(sess.run(t))
"""

