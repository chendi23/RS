import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import os
if os.environ.get('https_proxy'):
    channel = grpc.insecure_channel('192.168.3.135:80')
stub = prediction_service_pb2.PredictionServiceStub(channel)

# 模型签名
request = predict_pb2.PredictRequest()
request.model_spec.name = 'pb_mmoe'
#request.model_spec.version = 'latest'
request.model_spec.signature_name = 'predict'

# 构造入参
u_type = 2
u_age = 10
u_sex = 0
u_pos_id = 2
u_org_id = 3
u_seat_id = 0
i_class_label = 5


request.inputs['u_type'].CopyFrom(tf.make_tensor_proto(u_type, dtype=tf.int32))
request.inputs['u_age'].CopyFrom(tf.make_tensor_proto(u_age, dtype=tf.int32))
request.inputs['u_sex'].CopyFrom(tf.make_tensor_proto(u_sex, dtype=tf.float32))
request.inputs['u_org_id'].CopyFrom(tf.make_tensor_proto(u_org_id, dtype=tf.int32))
request.inputs['u_seat_id'].CopyFrom(tf.make_tensor_proto(u_seat_id, dtype=tf.int32))
request.inputs['u_pos_id'].CopyFrom(tf.make_tensor_proto(u_pos_id, dtype=tf.int32))
request.inputs['i_class_label'].CopyFrom(tf.make_tensor_proto(i_class_label, dtype=tf.int32))

#  返回CRF结果，输出发射概率矩阵和状态转移概率矩阵
result = stub.Predict(request, 10.0)  # 10 secs timeout

print(result)