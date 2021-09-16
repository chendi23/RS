import tensorflow as tf
import numpy as np
from tensorflow.contrib import  predictor
pb_path = 'pb_models'
predict_fn = predictor.from_saved_model(pb_path)


def gen_data():
    Xi , Xv= [], []
    for i in range(50):
        pick = np.random.choice(np.arange(200), 20, replace=False)
        sorted_pick = np.sort(pick)
        Xi.append(sorted_pick)
        Xv.append(np.random.randint(0, 2, 20))
    y = np.random.randint(0, 2, 50).reshape([50,1])

    return Xi, Xv, y

Xi,Xv,_ = gen_data()


def input_fn():
    feed_dict = {'X1': Xi, 'Xv': Xv, 'dropout_keep_deep':[0.5,0.5,0.5] , 'dropout_keep_fm':[1.0,1.0]}
    return feed_dict

pred = predict_fn(input_fn())
print(pred)