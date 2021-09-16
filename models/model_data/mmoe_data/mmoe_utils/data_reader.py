import pandas as pd
import numpy as np
from copy import deepcopy
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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

df_reader.data_clean()
df_reader.get_category()
df_reader.get_input()
#print(df_reader.get_category())

#print(df_reader.user_df[:2])
#print(df_reader.user_df.head())
#print(df_reader.item_df[df_reader.item_cate_columns])

#df_reader.get_input()