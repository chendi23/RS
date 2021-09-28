import sys
import tensorflow as tf
import data_loader
import time
from sklearn.metrics import roc_auc_score

def model_early_stop(valid_metric_list, backstep_num):
    length = len(valid_metric_list)
    best_metric_score = max(valid_metric_list)
    if length > 3:
        backstep_count = 0
        for i in range(backstep_num):
            if valid_metric_list[-1 * (i + 1)] < best_metric_score:
                backstep_count += 1
                if backstep_count == backstep_num:
                    return 1
        return 0


def model_fit(model, params, train_file, predict_file):
    valid_metric_list = []
    for ep in range(params.epoch):
        begin_time = time.time()
        model.train(input_fn=lambda: data_loader.input_fn(train_file, params))
        results = model.evaluate(input_fn=lambda: data_loader.input_fn(predict_file, params))
        end_time = time.time()
        print('epoch: ', ep, 'eval score:', results['auc_metric'], 'loss:', results['loss'], 'train plus eval time:', end_time-begin_time)
        sys.stdout.flush()

        valid_metric_list.append(results['auc_metric'])

        if model_early_stop(valid_metric_list, backstep_num=3):
            print('training early stops!!!')
            trained_model_path = model_save_pb(params, model)
            return trained_model_path

    print('saved model_pb')
    trained_model_path = model_save_pb(params, model)
    return trained_model_path

def model_save_pb(params, model):
    """
        保存模型为tf-serving使用的pb格式
    """
    input_spec = {'Xi': tf.placeholder(shape=[None, params.field_size], dtype=tf.float32, name='Xi'),
                  'Xv': tf.placeholder(shape=[None, params.field_size], dtype=tf.float32, name='Xv')}

    model_input_receiving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=input_spec)
    return model.export_savedmodel(params.model_pb, model_input_receiving_fn)


def model_predict(trained_model_path, predict_file, params):
    """
        加载pb模型,预测tfrecord类型的数据
    """
    with tf.Session(graph=tf.Graph()) as sess:
        model = tf.saved_model.loader.load(trained_model_path)
        features_dict, labels_dict = data_loader.input_fn(predict_file, params)
        while True:
            feature1, feature2, label1 = sess.run(features_dict['feature1'], features_dict['feature2'], labels_dict['label1'])
            feed_dict = {'feature1:0': feature1, 'feature2:0': feature2}
            score_list = []
            label_list = []

            prediction = sess.run('score:0', feed_dict=feed_dict)
            prediction_score = prediction[:, 0]
            label = label1[:,0]
            score_list.extend(prediction_score)
            label_list.extend(label)
            prediction_res = roc_auc_score(prediction, label1)
            print(prediction_res(prediction_res))

