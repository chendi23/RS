import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
#tf.enable_eager_execution()

favorite_book = [name.encode('utf-8') for name in ['cv', 'nlp']]
favorite_book_bytelist = tf.train.BytesList(value=favorite_book)
print('favorite_book_bytelist: ', favorite_book_bytelist)

working_hours = [1.0, 2.0, 3.0, 4.0, 5.0]
working_hours_floatlist = tf.train.FloatList(value=working_hours)
print('working_hours_floatlist: ', working_hours_floatlist)


age = [24]
age_int64_list = tf.train.Int64List(value=age)
print('age_int64_list: ', age_int64_list)

features = tf.train.Features(
    feature={
        'favorite_book': tf.train.Feature(bytes_list=favorite_book_bytelist),
        'hours': tf.train.Feature(float_list=working_hours_floatlist),
        'age':tf.train.Feature(int64_list=age_int64_list)
    }
)

example = tf.train.Example(features=features)

serialized_example = example.SerializeToString()

output_dir = os.path.join('tfrecord_basis')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

filename = 'test.tfrecords'
filename_full_path = os.path.join(output_dir, filename)
with tf.io.TFRecordWriter(filename_full_path) as wr:
    for i in range(2):
        #print(serialized_example)
        wr.write(serialized_example)



### load TFRecord with the same feature protocol
expected_features = {
    'favorite_books':tf.io.VarLenFeature(dtype=tf.string),
    'hours': tf.io.VarLenFeature(dtype=tf.float32),
    'age': tf.io.FixedLenFeature([], dtype=tf.int64)
}



"""
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
    books = tf.sparse.to_dense(example["favorite_books"])
    for book in books:
        print(book)

"""

ctr_frame = pd.read_csv('C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\ctr.csv', sep=';')
ctr_frame['user_id'] = LabelEncoder().fit_transform(y=ctr_frame['user_id'])
#print(ctr_frame['user_id'])
rows_count = len(ctr_frame)

output_tfrecord_dir = 'C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\ctr.tfrecords'



def get_ByteFeature(value):
    value_encode = value.encode('utf-8')
    value_encode_list = [value_encode]
    byte_list = tf.train.BytesList(value=value_encode_list)
    return tf.train.Feature(bytes_list=byte_list)

def get_FloatFeature(value):
    value_list = [value]
    float_list = tf.train.FloatList(value=value_list)
    return tf.train.Feature(float_list=float_list)

def get_IntFeature(value):
    value_list = [value]
    int_list = tf.train.Int64List(value=value_list)
    return tf.train.Feature(int64_list=int_list)


### write csv to tfrecord
"""
with tf.io.TFRecordWriter(output_tfrecord_dir) as wr:
    for i in range(rows_count):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'user_id':get_ByteFeature(ctr_frame.loc[i, 'user_id']),
                    'item_id':get_ByteFeature(ctr_frame.loc[i, 'item_id']),
                    'label': get_IntFeature(ctr_frame.loc[i, 'click'])
                }
            )
               )
        wr.write(record=example.SerializeToString())
    wr.close()
"""
### parse tfrecord

def parse_tfrecord():
    expected_features = {'user_id': tf.io.FixedLenFeature([1], dtype=tf.string),
                         'item_id': tf.io.FixedLenFeature([1], dtype=tf.string),
                         'label': tf.io.FixedLenFeature([], dtype=tf.int64)}

    dataset = tf.data.TFRecordDataset([output_tfrecord_dir])
    for serialized_example in dataset:
        example = tf.io.parse_single_example(serialized_example, expected_features)

    return



def parse_csv_line(line, n_fields=1):
    defs = [tf.constant(np.nan)]*n_fields
    parsed_fields = tf.io.decode_csv(line, select_cols=[3], record_defaults=defs, field_delim=';')
    x = tf.stack(parsed_fields[-1])
    y = tf.stack(parsed_fields[-1])
    #print('x:',x,'y:',y)
    return x,y


def get_csv_tf_dataset(file_dir):
    dataset = tf.data.Dataset.list_files(file_dir)
    dataset = dataset.repeat()
    dataset = dataset.interleave(lambda x:tf.data.TextLineDataset(x).skip(1), cycle_length=4)
    dataset.shuffle(10000)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=1)
    dataset = dataset.batch(batch_size=32)
    return dataset

dataset = get_csv_tf_dataset(['C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\ctr\\ctr.csv'])
print(tf.data.get_output_shapes(dataset))
iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()
with tf.Session() as sess:
    print(sess.run(x))


df = pd.read_csv('C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\merged_df_i_class_label.csv')
print(df.columns)
cate_column = ['position_id', 'org_id', 'seat_id', 'sex', 'user_type','i_class_label']
columns = ['position_id', 'org_id', 'seat_id', 'sex', 'user_type','age','i_class_label','click']
for cate in cate_column:
    le = LabelEncoder()
    classes = le.fit_transform(df[cate])
    df[cate] = classes
df = df[columns]
print(df.head())

"""
rows_count = df.shape[0]
with tf.io.TFRecordWriter(path='./merged_df.tfrecords') as wr:
    for i in range(rows_count):
        example = tf.train.Features(
            feature={
                'user_type': get_FloatFeature(df.loc[i, 'user_type']),
                'position_id':get_FloatFeature(df.loc[i,'position_id']),
                'org_id':get_FloatFeature(df.loc[i, 'org_id']),
                'seat_id': get_FloatFeature(df.loc[i, 'seat_id']),
                'sex': get_FloatFeature(df.loc[i, 'sex']),
                'age': get_FloatFeature(df.loc[i, 'age']),
                'i_class_label':get_IntFeature(df.loc[i,'i_class_label']),
                'click':get_IntFeature(df.loc[i,'click'])

            }
        )
        wr.write(example.SerializeToString())
    wr.close()


"""
