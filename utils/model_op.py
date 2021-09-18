import sys
import tensorflow as tf
import data_loader
import time


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
    for ep in params.epoch:
        begin_time = time.time()
        model.train(input_fn=lambda: data_loader.input_fn(train_file, params))
        results = model.evaluate(model_fn=lambda: data_loader.input_fn((predict_file, params)))
        end_time = time.time()
        print('epoch: ', ep, 'eval score:', results['auc_metric'], 'loss:', results['loss'], 'train plus eval time:', end_time-begin_time)
        sys.stdout.flush()

        valid_metric_list.append(results['auc_metric'])

        if model_early_stop(valid_metric_list, backstep_num=1):
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
    input_spec = {'feature1': tf.placeholder(shape=[None, params.feature1_size], dtype=tf.float32, name='feature1'),
                  'feature2': tf.placeholder(shape=[None, params.feature2_size], dtype=tf.float32, nmae='feature2')}

    model_input_receiving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=input_spec)
    return model.export_savedmodel(params.model_pb, model_input_receiving_fn)


def model_predict(trained_model_path, predict_file, params):
    """
        加载pb模型,预测tfrecord类型的数据
    """
    with tf.Session(graph=tf.Graph()):
        model = tf.saved_model.loader.load(trained_model_path)
        features_dict, labels_dict = data_loader.input_fn(predict_file, params)
        ###TO DO!!!


    return

