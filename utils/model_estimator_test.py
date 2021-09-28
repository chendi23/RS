import tensorflow as tf


def model_fn(labels, features, mode, params):
    tf.set_random_seed(2021)  # experiment reproducibility

    with tf.name_scope('inputs'):
        Xi = tf.to_int32(features['Xi'])
        Xv = features['Xv']


    with tf.name_scope('embedding_fc_layer'):
        Xi_emb_matrix = tf.Variable(initial_value=tf.random_normal(shape=[params.feature_dim, params.emb_dim]), dtype=tf.float32)
        Xi_emb_layer = tf.nn.embedding_lookup(Xi_emb_matrix, Xi)
        Xv_reshape = tf.reshape(Xv, shape=[-1, params.field_size, 1])
        embeddings = tf.multiply(Xi_emb_layer, Xv_reshape)
        embeddings = tf.reshape(embeddings, shape=[-1, params.field_size*params.emb_dim])


    with tf.name_scope('outputs'):
        output = tf.layers.dense(embeddings, 1)

        sigmoid_out = tf.sigmoid(output)
        score = tf.identity(sigmoid_out, name='score')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=score)

    else:
        label1 = tf.identity(tf.reshape(labels, [-1, 1]), name='label1')
        #label1 = labels
        with tf.name_scope('metrics'):
            auc_metric = tf.metrics.auc(labels=label1, predictions=score, name='auc_metric')

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.losses.log_loss(labels=label1, predictions=score), name='mse_loss')

        metrics = {'auc_metric': auc_metric}
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.lr)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, train_op=train_op)


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

