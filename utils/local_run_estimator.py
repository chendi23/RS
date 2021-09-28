import data_loader
import model_estimator_test
import model_op
import argparse


def arg_parser():
    params = argparse.ArgumentParser()
    params.add_argument('--lr', default=0.01)
    params.add_argument('--batch_size', default=32)
    params.add_argument('--epoch', default=10)
    params.add_argument('--mode', default='train')
    params.add_argument('--feature_dim', default=106)
    params.add_argument('--field_size', default=10)
    params.add_argument('--emb_dim', default=32)
    params.add_argument('--model_dir', default='D:\\zcd\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\utils\\model_dir')
    params.add_argument('--model_pb', default='D:\\zcd\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\utils\\model_pb')
    params.add_argument('--train_path', default='D:\\zcd\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\utils\\tf_record_from_lists')
    params.add_argument('--predict_path', default='D:\\zcd\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\utils\\tf_record_from_lists')
    params.add_argument('--log_step_count_steps', default=100000)
    params.add_argument('--save_checkpoints_steps', default=100000)
    params.add_argument('--keep_checkpoint_max', default=100000)
    params.add_argument('--save_summary_steps', default=100000)
    params.add_argument('--is_GPU', default=0)
    return params.parse_args()


def main():
    params = arg_parser()
    model = model_estimator_test.model_estimator(params)
    if params.mode == 'train':
        train_files = data_loader.get_file_list(params.train_path)
        predict_files = data_loader.get_file_list(params.predict_path)
        model_op.model_fit(model=model, params=params, train_file=train_files, predict_file=predict_files)
    elif params.mode == 'predict':
        predict_file = data_loader.get_file_list(params.predict_path)
        model_op.model_predict(trained_model_path=params.model_dir, predict_file=predict_file, params=params)


if __name__ == '__main__':
    main()
