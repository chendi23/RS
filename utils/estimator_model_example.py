import tensorflow as tf


def model_fn(labels, features, mode, params):
    tf.set_random_seed(2021)  # experiment reproducibility

    with tf.name_scope('inputs'):
        input_feature_1 = features['feature1']
        input_feature_2 = features['feature2']

    with tf.name_scope('outputs'):
        output = tf.multiply(input_feature_1, input_feature_2, name='raw')
        score = tf.identity(tf.sigmoid(output), name='score')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=score)

    else:
        label1 = tf.identity(labels['label1'], name='label1')

        with tf.name_scope('metrics'):
            metric = tf.metrics.auc(score, label1, name='auc_metric')

        with tf.name_scope('loss'):
            mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(score, label1), name='mse loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.lr)
        train_op = optimizer.minimize(loss=mse_loss, global_step=tf.train.get_global_step())
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(mode=mode, loss=mse_loss, eval_metric_ops=metric, train_op=train_op)


def model_estimator(params):
    tf.reset_default_graph()
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': params.is_GPU}),
        log_step_count_steps=params.log_step_count_steps,
        save_checkpoints_steps=params.save_checkpoints_steps,
        keep_checkpoint_max=params.keep_checkpoint_max,
        save_summary_steps=params.save_summary_steps
    )

    model = tf.estimator.Estimator(model_fn, config=config, model_dir=params.model_dir, params=params)

    return model

