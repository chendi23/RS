import pandas as pd
import numpy as np
from copy import deepcopy
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import os
from datetime import datetime
#tf.enable_eager_execution()

class DataReader:
    def __init__(self, u_csv_path, i_csv_path, ctr_csv_path, ratings_csv_path):
        self.user_df = pd.read_csv(u_csv_path, sep=';')
        self.item_df = pd.read_csv(i_csv_path, sep=';')
        self.ctr_df = pd.read_csv(ctr_csv_path, sep=';')
        self.ratings_df = pd.read_csv(ratings_csv_path, sep=';')
        self.user_cate_columns = ['user_id', 'user_type', 'sex', 'org_id', 'seat_id', 'position_id', 'age']
        self.item_cate_columns = ['i_class_label']
        self.bins = {'age': 10}

    def data_clean(self):
        u_dict, item_dict = {}, {}
        u_df = deepcopy(self.user_df)
        i_df = deepcopy(self.item_df)

        # clean the user dataframe
        self.user_df = u_df[u_df['age'].notna()]

        # clean the item dataframe
        for i in range(len(i_df)):
            self.item_df.at[i, 'i_class_label'] = i_df.at[i, 'i_class_label'].replace(' ','')
            self.item_df.at[i, 'i_entities_label'] = literal_eval(i_df.at[i, 'i_entities_label'].replace(u'\\u3000', '').replace(' ',''))
        self.item_df = self.item_df[self.item_df['i_class_label'].map(len) > 0]
        self.item_df = self.item_df[self.item_df['i_entities_label'].map(len) > 0]
        self.item_df = self.item_df[['item_id', 'i_class_label', 'i_entities_label']]

        # reindex with len(df)
        self.item_df = self.item_df.reset_index(drop=True)
        self.user_df = self.user_df.reset_index(drop=True)

        print(len(self.item_df))

    def data_clean_retain_str(self):
        u_dict, item_dict = {}, {}
        u_df = deepcopy(self.user_df)
        i_df = deepcopy(self.item_df)

        # clean the user dataframe
        self.user_df = u_df[u_df['age'].notna()]

        # clean the item dataframe
        for i in range(len(i_df)):
            self.item_df.at[i, 'i_class_label'] = i_df.at[i, 'i_class_label'].replace(' ','')
            self.item_df.at[i, 'i_entities_label'] = literal_eval(i_df.at[i, 'i_entities_label'].replace(u'\\u3000', '').replace(' ',''))
        self.item_df = self.item_df[self.item_df['i_class_label'].map(len) > 0]
        self.item_df = self.item_df[self.item_df['i_entities_label'].map(len) > 0]
        self.item_df = self.item_df[['item_id', 'i_class_label', 'i_entities_label']]

        # reindex with len(df)
        self.item_df = self.item_df.reset_index(drop=True)
        self.user_df = self.user_df.reset_index(drop=True)

        print(len(self.item_df))
    def get_category(self):

        cate_num_dict = {}
        age_cuts = pd.cut(x=self.user_df['age'], bins=self.bins['age'])
        self.user_df['age'] = age_cuts
        for i in self.user_cate_columns:
            le = LabelEncoder()
            labels = le.fit_transform(self.user_df[i])
            self.user_df[i] = labels
            cate_num_dict[i] = len(np.unique(labels))

        for j in self.item_cate_columns:
            le = LabelEncoder()
            labels = le.fit_transform(self.item_df[j])
            self.item_df[j] = labels
            cate_num_dict[j] = len(np.unique(labels))

        return cate_num_dict

    def get_df_input(self):
        cate_num_dict = {}


        ctr_plus_item = pd.merge(left=self.ctr_df, right=self.item_df[['item_id']], on=['item_id'], how='inner')
        #print(ctr_plus_item.columns)
        user_filtered = pd.merge(left=ctr_plus_item, right=self.user_df.drop(columns=['user_type']), on=['user_id'], how='inner')
        #print(user_filtered.columns)
        item_filtered = pd.merge(left=user_filtered, right=self.item_df, on=['item_id'], how='inner')
        #print(item_filtered.columns)
        click_filterd = pd.merge(left=item_filtered, right=self.ctr_df.drop(columns=['user_type', 'click']), on = ['user_id','item_id'], how='inner')
        #print(click_filterd.columns)
        click_filterd['id'] = click_filterd['user_id'] + '|' + click_filterd['item_id']
        click_filterd.drop(columns=['user_id','item_id'], inplace=True)
        #print(click_filterd.columns)
        return click_filterd

        return

    def get_input(self, wv_array='C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\items_and_wv_np.npy'):
        """
        wv_emb_csv: a .noy file with word vectors
        """

        u_type = np.zeros([len(self.ctr_df), 1])
        u_sex = np.zeros([len(self.ctr_df), 1])
        u_age = np.zeros([len(self.ctr_df), 1])
        u_pos = np.zeros([len(self.ctr_df), 1])
        u_seat = np.zeros([len(self.ctr_df), 1])
        u_org = np.zeros([len(self.ctr_df), 1])
        i_class = np.zeros([len(self.ctr_df), 1])
        i_entities = np.zeros([len(self.ctr_df), 1024])
        label1 = np.zeros([len(self.ctr_df), 1])
        label2 = np.zeros([len(self.ctr_df), 1])
        if wv_array is not None:
            wv_array = np.load(wv_array)
            for i in range(len(self.ctr_df)):
                # print(self.ctr_df.loc[i, 'user_id'], self.ctr_df.loc[i,'item_id']za)
                if self.ctr_df.loc[i, 'user_id'] in self.user_df['user_id'].to_list() and self.ctr_df.loc[i, 'item_id'] in self.item_df['item_id'].to_list():
                    user_index = pd.Index(self.user_df.loc[:, 'user_id']).get_loc(self.ctr_df.loc[i, 'user_id'])
                    item_index = pd.Index(self.item_df.loc[:, 'item_id']).get_loc(self.ctr_df.loc[i, 'item_id'])

                    u_type[i] = self.user_df.at[user_index, 'user_type']
                    u_sex[i] = self.user_df.at[user_index, 'sex']
                    u_age[i] = self.user_df.at[user_index, 'age']
                    u_pos[i] = self.user_df.at[user_index, 'position_id']
                    u_seat[i] = self.user_df.at[user_index, 'seat_id']
                    u_org[i] = self.user_df.at[user_index, 'org_id']
                    i_class[i] = self.item_df.at[item_index, 'i_class_label']
                    i_entities[i] = wv_array[item_index,:]

                    label1[i] = self.ctr_df.at[i, 'click']
                    label2[i] = self.ctr_df.at[i, 'click']
        else:
            for i in range(len(self.ctr_df)):
                # print(self.ctr_df.loc[i, 'user_id'], self.ctr_df.loc[i,'item_id'])
                if self.ctr_df.loc[i, 'user_id'] in self.user_df['user_id'].to_list() and self.ctr_df.loc[i, 'item_id'] in self.item_df['item_id'].to_list():
                    user_index = pd.Index(self.user_df.loc[:, 'user_id']).get_loc(self.ctr_df.loc[i, 'user_id'])
                    item_index = pd.Index(self.item_df.loc[:, 'item_id']).get_loc(self.ctr_df.loc[i, 'item_id'])

                    u_type[i] = self.user_df.at[user_index, 'user_type']
                    u_sex[i] = self.user_df.at[user_index, 'sex']
                    u_age[i] = self.user_df.at[user_index, 'age']
                    u_pos[i] = self.user_df.at[user_index, 'position_id']
                    u_seat[i] = self.user_df.at[user_index, 'seat_id']
                    u_org[i] = self.user_df.at[user_index, 'org_id']
                    i_class[i] = self.item_df.at[item_index, 'i_class_label']

                    label1[i] = self.ctr_df.at[i, 'click']
                    label2[i] = self.ctr_df.at[i, 'click']





        print(len(u_type), len(i_class))
        return u_type, u_sex, u_org, u_age, u_pos, u_seat, i_class, i_entities, label1, label2



df_reader = DataReader(i_csv_path='C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\items.csv',
                                   u_csv_path='C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\users.csv',
                                   ctr_csv_path='C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\ctr.csv',
                                   ratings_csv_path='C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\ratings.csv')

df_reader.data_clean_retain_str()
#df_reader.get_category()
df_input = df_reader.get_df_input()
df_input.to_csv('./processed_csv', sep=';')
#df_reader.get_input()
#print(df_reader.get_category())

#print(df_reader.user_df[:2])
#print(df_reader.user_df.head())
#print(df_reader.item_df[df_reader.item_cate_columns])

#df_reader.get_input()



class FeatureDictionary:
    def __init__(self, df=None, numeric_cols=['age'], ignore_cols=['click', 'heat', 'grade_id','grade_name','u_class_label','i_entities_label','user_name', 'id','u_entities_label','i_keywords_label', 'u_keywords_label']):
        assert not(df is None)
        self.df = df
        self.rows_count = self.df.shape[0]
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        #self.gen_feature_dictionary()

    def gen_feature_dictionary(self):
        feature_dict = {}
        col_count = 0
        for col in self.df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                feature_dict[col] = col_count
                col_count += 1
            else:
                us = self.df[col].unique()
                feature_dict[col] = dict(zip(us, range(col_count, col_count+len(us))))
                col_count += len(us)
        self.feature_dim = col_count
        print(feature_dict)

        return feature_dict

fd_object = FeatureDictionary(df=df_input)


class DataParser:
    def __init__(self, feature_dict_ob):
        assert isinstance(feature_dict_ob, FeatureDictionary)
        self.feature_dict_ob = feature_dict_ob
        self.feature_dict = self.feature_dict_ob.gen_feature_dictionary()
        self.feature_dim = self.feature_dict_ob.feature_dim
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


ps = DataParser(fd_object)
feature_dim, rows_count = ps.feature_dim, ps.rows_count
Xi, Xv, labels = ps.parse(df_input)
lists_dict = {'Xi':Xi, 'Xv':Xv, 'labels': labels}



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

def get_LabelFeature(value):
    value = [value]
    float_list = tf.train.FloatList(value=value)
    return tf.train.Feature(float_list)



def list_to_tfrecords(lists_dict=None):
    assert not (lists_dict is None)
    output_dir = 'tf_record_from_lists'
    if not os.path.exists(os.path.join(output_dir)):
        os.mkdir(output_dir)
    filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S' ) + '.tfrecords'
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

#list_to_tfrecords(lists_dict)


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
        dataset = dataset.map(parse_example)
        print(tf.data.get_output_shapes(dataset))

    return

"""
expected_features = {}
expected_features['Xi'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
expected_features['Xv'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
expected_features['labels'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

dataset = tf.data.TFRecordDataset(['.//tf_record_from_lists//2021_09_17_19_12_56.tfrecords'])
for example in dataset:
    parsed_feature_dict = tf.io.parse_single_example(example, features=expected_features)
    print(parsed_feature_dict)
"""
parse_tfrecords('.//tf_record_from_lists//2021_09_17_19_12_56.tfrecords')