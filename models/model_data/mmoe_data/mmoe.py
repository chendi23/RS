#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import models.model_data.mmoe_data.mmoe_utils.model_layer as my_layer
from tensorflow import initializers
from mmoe_utils import data_reader
import demo_data


class MMOE():
    def __init__(self,
                 epochs=100,
                 batch_size=128,
                 units=16,
                 num_experts=4,
                 num_tasks=2,
                 list_task_hidden_units=[16, 32],
                 u_emb_dims=16,
                 u_output_dims=64,
                 i_class_emb_dims=10,
                 i_wv_reduced_to_dims=8,
                 i_output_dims=16,
                 is_using_wv=True,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None
                 ):

        # embedding parameters
        self.is_using_wv = is_using_wv
        self.u_emb_dims = u_emb_dims
        self.u_output_dims = u_output_dims
        self.i_class_emb_dims = i_class_emb_dims
        self.i_wv_reduced_to_dims = i_wv_reduced_to_dims
        self.i_out_dims = i_output_dims

        # hidden nodes parameters
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.units = units
        self.list_task_hidden_units = list_task_hidden_units

        # weight parameters+-
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.variance_scaling()
        self.gate_kernel_initializer = initializers.variance_scaling()
        self.expert_kernel_regularizer = expert_kernel_regularizer
        self.gate_kernel_regularizer = gate_kernel_regularizer
        self.expert_kernel_constraint = expert_kernel_constraint
        self.gate_kernel_constraint = gate_kernel_constraint

        # activity paramaters
        self.activity_regularizer = activity_regularizer

        # bias parameters
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.zeros()
        self.gate_bias_initializer = initializers.zeros()
        self.expert_bias_regularizer = expert_bias_regularizer
        self.gate_bias_regularizer = gate_bias_regularizer
        self.expert_bias_constraint = expert_bias_constraint
        self.gate_bias_constraint = gate_bias_constraint

        # initialize the graph
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = tf.train.AdamOptimizer()
        self.sess = tf.Session()
        self._init_graph()

    def _init_graph(self):
        np.random.seed(2021)
        #self.graph = tf.Graph()
        #with self.graph.as_default():
        #tf.set_random_seed(2021)
        # user_info
        self.u_type = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='u_type')
        self.u_age = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='u_age')
        self.u_sex = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='u_sex')
        self.u_pos_id = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='u_pos_id')
        self.u_seat_id = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='u_seat_id')
        self.u_org_id = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='u_org_id')
        # item_info
        self.i_class_label = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='i_class_label')
        self.i_entities = tf.placeholder(dtype=tf.float32, shape=[None, 1024], name='i_entities')
        # labels
        self.ctr_label = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.ctrcvr_label = tf.placeholder(dtype=tf.int32, shape=[None, 1])

        with tf.name_scope('user_embedding'):
            # user type embedding
            self.u_type_emb_matrix = tf.Variable(initial_value=tf.random_uniform([cate_num_dict['user_type'], self.u_emb_dims], -1, 1), name='u_type_emb_matrix')

            self.u_type_emb_layer = tf.nn.embedding_lookup(self.u_type_emb_matrix, self.u_type, name='u_type_emb_layer')

            # user age embedding
            self.u_age_emb_matrix = tf.Variable(initial_value=tf.random_uniform([cate_num_dict['age'], self.u_emb_dims], -1, 1),name='u_age_emn_matrix')

            self.u_age_emb_layer = tf.nn.embedding_lookup(self.u_age_emb_matrix, self.u_age, name='u_age_emb_layer')

            # user sex embedding
            self.u_sex_emb_matrix = tf.Variable(initial_value=tf.random_uniform([cate_num_dict['age'], self.u_emb_dims], -1, 1),name='u_sex_emb_matrix')

            self.u_sex_emb_layer = tf.nn.embedding_lookup(self.u_sex_emb_matrix, self.u_sex, name='u_sex_emb_layer')

            # user org_id embedding
            self.u_org_emb_matrix = tf.Variable(initial_value=tf.random_uniform([cate_num_dict['org_id'], self.u_emb_dims], -1, 1),name='u_org_emb_matrix')

            self.u_org_emb_layer = tf.nn.embedding_lookup(self.u_org_emb_matrix, self.u_org_id, name='u_org_emb_layer')

            # user seat_id embedding

            self.u_seat_emb_matrix = tf.Variable(initial_value=tf.random_uniform([cate_num_dict['seat_id'], self.u_emb_dims], -1, 1), name='u_seat_emb_matrix')

            self.u_seat_emb_layer = tf.nn.embedding_lookup(self.u_seat_emb_matrix, self.u_seat_id, name='u_seat_emb_layer')

            # user pos_id embedding

            self.u_pos_emb_matrix = tf.Variable(initial_value=tf.random_uniform([cate_num_dict['position_id'], self.u_emb_dims], -1, 1),name='u_pos_emb_matrix')

            self.u_pos_emb_layer = tf.nn.embedding_lookup(self.u_pos_emb_matrix, self.u_pos_id, name='u_pos_emb_layer')

        with tf.name_scope('user_fc'):
            self.u_type_fc = tf.layers.dense(self.u_type_emb_layer, self.u_emb_dims, name='u_type_fc',
                                             activation=tf.nn.relu,
                                             bias_regularizer=None,
                                             kernel_regularizer=None)

            self.u_age_fc = tf.layers.dense(self.u_age_emb_layer, self.u_emb_dims, name='u_age_fc',
                                            activation=tf.nn.relu,
                                            bias_regularizer=None,
                                            kernel_regularizer=None)
            self.u_sex_fc = tf.layers.dense(self.u_sex_emb_layer, self.u_emb_dims, name='u_sex_fc',
                                            activation=tf.nn.relu,
                                            bias_regularizer=None)

            self.u_org_fc = tf.layers.dense(self.u_org_emb_layer, self.u_emb_dims, name='u_org_fc',
                                            activation=tf.nn.relu,
                                            bias_regularizer=None,
                                            kernel_regularizer=None)

            self.u_seat_fc = tf.layers.dense(self.u_seat_emb_layer, self.u_emb_dims, name='u_seat_fc',
                                             activation=tf.nn.relu,
                                             bias_regularizer=None,
                                             kernel_regularizer=None)

            self.u_pos_fc = tf.layers.dense(self.u_pos_emb_layer, self.u_emb_dims, name='u_pos_id',
                                            activation=tf.nn.relu,
                                            bias_regularizer=None,
                                            kernel_regularizer=None)

            self.u_concated = tf.concat(
                [self.u_type_fc, self.u_age_fc, self.u_sex_fc, self.u_org_fc, self.u_seat_fc, self.u_pos_fc], axis=2,
                name='u_concated')

            self.u_concated_fc = tf.layers.dense(self.u_concated, self.u_output_dims, name='u_concated_fc',
                                                 activation=tf.nn.relu,
                                                 bias_regularizer=None,
                                                 kernel_regularizer=None)


        with tf.name_scope('item_class_embedding'):
            self.i_class_emb_matrix = tf.Variable(initial_value=tf.random_uniform([cate_num_dict['i_class_label'], self.i_class_emb_dims], -1, 1), name='i_class_emb_matrix')

            self.i_class_emb_layer = tf.nn.embedding_lookup(self.i_class_emb_matrix, self.i_class_label,
                                                            name='i_class_emb_layer')

        with tf.name_scope('item_class_fc'):
            self.i_class_fc = tf.layers.dense(self.i_class_emb_layer, self.i_class_emb_dims, name='i_class_fc',
                                              activation=tf.nn.relu,
                                              bias_regularizer=None,
                                              kernel_regularizer=None)

        with tf.name_scope('item_wv_dim_reduce'):
            self.i_entities_emb_fc = tf.layers.dense(tf.expand_dims(self.i_entities, axis=1), self.i_wv_reduced_to_dims,
                                                     name='i_entities_emb_fc',
                                                     activation=tf.nn.relu,
                                                     bias_regularizer=None,
                                                     kernel_regularizer=None)

        with tf.name_scope('item_concated'):
            if self.is_using_wv:
                self.i_emb_concated = tf.concat([self.i_entities_emb_fc, self.i_class_fc], axis=2,
                                                name='i_emb_concated')
            else:
                self.i_emb_concated = self.i_class_fc

        with tf.name_scope('item_concated_fc'):
            self.i_concated_fc = tf.layers.dense(self.i_emb_concated, self.i_out_dims, name='i_concated_fc',
                                                 activation=tf.nn.relu,
                                                 bias_regularizer=None,
                                                 kernel_regularizer=None)

        with tf.name_scope('hidden'):
            self.hidden = tf.concat([self.u_concated_fc, self.i_concated_fc], axis=2)
            self.hidden = tf.reshape(self.hidden, shape=[-1, self.hidden.get_shape()[-1]])

        with tf.name_scope('expert'):
            self.expert_weight = tf.get_variable(dtype=tf.float32,
                                                 shape=(self.hidden.get_shape()[1], self.units, self.num_experts),
                                                 regularizer=self.expert_kernel_regularizer,
                                                 initializer=self.expert_kernel_initializer,
                                                 name='expert_weight')

            self.expert_bias = tf.get_variable(dtype=tf.float32,
                                               shape=(self.num_experts,),
                                               regularizer=self.expert_bias_regularizer,
                                               initializer=self.expert_bias_initializer,
                                               name='expert_bias')
            self.expert_out = tf.tensordot(self.hidden, self.expert_weight, axes=1)
            if self.use_expert_bias:
                self.expert_out = tf.add(self.expert_out, self.expert_bias)
            self.expert_out = tf.nn.relu(self.expert_out, name='expert_out')

        with tf.name_scope('gate1'):
            self.gate1_weight = tf.get_variable(shape=[self.hidden.get_shape()[1], self.num_experts],
                                                initializer=self.gate_kernel_initializer,
                                                regularizer=self.gate_kernel_regularizer,
                                                name='gate1_weight')

            self.gate1_bias = tf.get_variable(shape=(self.num_experts,),
                                              initializer=self.gate_bias_initializer,
                                              regularizer=self.gate_bias_regularizer,
                                              constraint=self.gate_bias_constraint,
                                              name='gate1_bias')
            self.gate1_out = tf.matmul(self.hidden, self.gate1_weight)
            if self.use_gate_bias:
                self.gate1_out = tf.add(self.gate1_out, self.gate1_bias)
            self.gate1_out = tf.nn.softmax(self.gate1_out, name='gate1_out')

        with tf.name_scope('gate2'):
            self.gate2_weight = tf.get_variable(shape=[self.hidden.get_shape()[1], self.num_experts],
                                                initializer=self.gate_kernel_initializer,
                                                regularizer=self.gate_kernel_regularizer,
                                                name='gate2_weight')

            self.gate2_bias = tf.get_variable(shape=(self.num_experts,),
                                              initializer=self.gate_bias_initializer,
                                              regularizer=self.gate_bias_regularizer,
                                              constraint=self.gate_bias_constraint,
                                              name='gate2_bias')
            self.gate2_out = tf.matmul(self.hidden, self.gate2_weight)
            if self.use_gate_bias:
                self.gate2_out = tf.add(self.gate2_out, self.gate2_bias)
            self.gate2_out = tf.nn.softmax(self.gate2_out, name='gate2_out')

        len_task_hidden_units = len(self.list_task_hidden_units)

        with tf.name_scope('label1_input'):
            self.label1_input = tf.multiply(self.expert_out, tf.expand_dims(self.gate1_out, 1))
            self.label1_input = tf.reduce_sum(self.label1_input, axis=2)
            self.label1_input = tf.reshape(self.label1_input, [-1, self.units])

        with tf.name_scope('label1_output'):
            self.ctr_layer = tf.layers.dense(inputs=self.label1_input, units=self.list_task_hidden_units[0],
                                             activation=tf.nn.relu)
            for i in range(1, len_task_hidden_units):
                self.ctr_layer = tf.layers.dense(inputs=self.ctr_layer, units=self.list_task_hidden_units[i],
                                                 activation=tf.nn.relu)
            self.ctr_out = tf.layers.dense(inputs=self.ctr_layer, units=1)
        self.ctr_score = tf.identity(tf.nn.sigmoid(self.ctr_out), name='ctr')

        with tf.name_scope('label2_input'):
            self.label2_input = tf.multiply(self.expert_out, tf.expand_dims(self.gate2_out, axis=1))
            self.label2_input = tf.reduce_sum(self.label2_input, axis=2)
            self.label2_input = tf.reshape(self.label2_input, [-1, self.units])

        with tf.name_scope('label2_output'):
            self.ctrcvr_layer = tf.layers.dense(inputs=self.label2_input, units=self.list_task_hidden_units[0],
                                                activation=tf.nn.relu)
            for i in range(1, len_task_hidden_units):
                self.ctr_layer = tf.layers.dense(inputs=self.ctrcvr_layer, units=self.list_task_hidden_units[i],
                                                 activation=tf.nn.relu)
            self.ctrcvr_out = tf.layers.dense(inputs=self.ctrcvr_layer, units=1)
        self.ctrcvr_score = tf.identity(tf.nn.sigmoid(self.ctrcvr_out), name='cvr_ctr')

        with tf.name_scope('loss'):
            self.ctr_loss = tf.reduce_mean(tf.losses.log_loss(self.ctr_score, predictions=self.ctr_label))
            self.ctrcvr_loss = tf.reduce_mean(tf.losses.log_loss(self.ctrcvr_score, predictions=self.ctr_label))
            self.loss = self.ctr_loss + self.ctrcvr_loss

    def get_batch(self, u_type, u_sex, u_org, u_age, u_pos, u_seat, i_class, i_entities, label1, label2, batch_size, index):
        start = index*batch_size
        end = (1+index)*batch_size
        end = end if end < len(label1) else len(label1)

        return u_type[start:end], u_sex[start:end], u_org[start:end], u_age[start:end], u_pos[start:end], u_seat[start:end], i_class[start:end], i_entities[start:end], label1[start:end], label2[start:end]

    def fit_on_batch(self, u_type_, u_sex_, u_org_, u_age_, u_pos_, u_seat_, i_class_, i_entities_, label1_, label2_):
        feed_dict = {self.u_type: u_type_,
                     self.u_sex: u_sex_,
                     self.u_org_id: u_org_,
                     self.u_age: u_age_,
                     self.u_pos_id: u_pos_,
                     self.u_seat_id: u_seat_,
                     self.i_class_label: i_class_,
                     self.i_entities: i_entities_,
                     self.ctr_label: label1_,
                     self.ctrcvr_label: label2_
                     }

        loss, ctr_gt, ctr_out, _ = self.sess.run([self.loss, self.ctr_label, self.ctr_score, self.train_op], feed_dict=feed_dict)
        ctr_pred = []
        for i in ctr_out:
            if i > 0.5:
                ctr_pred.append([1])
            else:
                ctr_pred.append([0])
        hits = sum(ctr_pred == ctr_gt)
        #print('hits： ', hits)
        #print('train batch loss: ', loss, 'train batch accuracy: ', hits/self.batch_size)

        return hits, loss

    def fit(self, u_type, u_sex, u_org, u_age, u_pos, u_seat, i_class, i_entities, label1, label2):
        train_rate = 0.5
        train_size = int(u_type.shape[0]*train_rate)
        total_batches = int(train_size/self.batch_size)
        total_batches_val = int((u_type.shape[0]-train_size)/self.batch_size)
        with self.sess:
            self.train_op = self.optimizer.minimize(self.loss)
            init = tf.initialize_all_variables()
            self.sess.run(init)
            for ep in range(self.epochs):
                for i in range(total_batches):
                    u_type_, u_sex_, u_org_, u_age_, u_pos_, u_seat_, i_class_, i_entities_, label1_, label2_ = self.get_batch(
                        u_type[:train_size,:], u_sex[:train_size,:],
                        u_org[:train_size,:], u_age[:train_size,:], u_pos[:train_size,:], u_seat[:train_size,:],
                        i_class[:train_size,:], i_entities[:train_size,:], label1[:train_size,:], label2[:train_size,:],
                        self.batch_size, index=i)
                    self.fit_on_batch(u_type_, u_sex_, u_org_, u_age_, u_pos_, u_seat_, i_class_, i_entities_, label1_,
                                      label2_)

                loss_, hits_ = self.predict(u_type[train_size:,:], u_sex[train_size:,:], u_org[train_size:,:], u_age[train_size:,:], u_pos[train_size:,:], u_seat[train_size:,:], i_class[train_size:,:],i_entities[train_size:,:], label1[train_size:,:], label2[train_size:,:])

                print('epoch index:', ep, 'eva loss: ', loss_, 'hits: ', hits_, 'ep acc', hits_/(u_type.shape[0]-train_size))

            """
            tf.saved_model.simple_save(export_dir='./pb_models', session=self.sess, inputs={'u_type': self.sess.graph.get_tensor_by_name('u_type:0'),
                                                                                                   'u_sex': self.sess.graph.get_tensor_by_name('u_sex:0'),
                                                                                                   'u_org': self.sess.graph.get_tensor_by_name('u_org_id:0'),
                                                                                                   'u_age': self.sess.graph.get_tensor_by_name('u_age:0'),
                                                                                                   'u_pos':self.sess.graph.get_tensor_by_name('u_pos_id:0'),
                                                                                                   'u_seat':self.sess.graph.get_tensor_by_name('u_seat_id:0'),
                                                                                                   'i_class_label':self.sess.graph.get_tensor_by_name('i_class_label:0'),
                                                                                                   'i_entities_label':self.sess.graph.get_tensor_by_name('i_entities:0')},
                                              outputs={'label1_score':self.sess.graph.get_tensor_by_name('ctr:0'),
                                                       'label2_score':self.sess.graph.get_tensor_by_name('cvr_ctr:0')})
            """
            tf.summary.FileWriter('./logs', tf.get_default_graph())



        return

    def predict(self, u_type, u_sex, u_org, u_age, u_pos, u_seat, i_class, i_entities, label1, label2):

        feed_dict = {self.u_type: u_type,
                     self.u_sex: u_sex,
                     self.u_org_id: u_org,
                     self.u_age: u_age,
                     self.u_pos_id: u_pos,
                     self.u_seat_id: u_seat,
                     self.i_class_label: i_class,
                     self.i_entities: i_entities,
                     self.ctr_label: label1,
                     self.ctrcvr_label: label2
                     }
        loss, ctr_out, ctr_gt = self.sess.run([self.loss, self.ctr_score, self.ctr_label],
                                                 feed_dict=feed_dict)
        ctr_pred = []
        for i in ctr_out:
            if i > 0.5:
                ctr_pred.append([1])
            else:
                ctr_pred.append([0])
        hits = sum(ctr_pred == ctr_gt)
        wrong_index = [i for i in range(len(u_type)) if ctr_pred[i] != ctr_gt[i]]
        wrong_index = [i+46489+2 for i in wrong_index]
        print('wrong index: ', wrong_index)
        return loss, hits

df_reader = data_reader.DataReader(i_csv_path='C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\items_hai.csv',
                                   u_csv_path='C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\users.csv',
                                   ctr_csv_path='C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\ctr.csv',
                                   ratings_csv_path='C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\ratings.csv')
df_reader.data_clean()
cate_num_dict = df_reader.get_category()


u_type, u_sex, u_org, u_age, u_pos, u_seat, i_class, i_entities, label1, label2 = df_reader.get_input('C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data\\items_and_wv_np.npy')
shuffle_train = np.random.choice(u_type.shape[0], u_type.shape[0], False)



mmoe = MMOE()
mmoe.fit(u_type[shuffle_train], u_sex[shuffle_train], u_org[shuffle_train], u_age[shuffle_train], u_pos[shuffle_train], u_seat[shuffle_train],
         i_class[shuffle_train], i_entities[shuffle_train], label1[shuffle_train], label2[shuffle_train])





