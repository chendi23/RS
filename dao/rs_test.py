import pandas as pd
import os
import logging as logger


def load_dataset(train_path_dir):
    users = pd.read_csv(os.path.join(train_path_dir, 'users.csv'),
                        usecols=['user_id', 'user_type',
                                 'org_id', 'seat_id', 'grade_id', 'position_id', 'sex', 'age',
                                 'u_keywords_label', 'u_class_label', 'u_entities_label'],
                        sep=';',
                        error_bad_lines=False,
                        encoding='utf-8')
    items = pd.read_csv(os.path.join(train_path_dir, 'items_hai.csv'),
                        usecols=['item_id', 'category_id', 'title', 'content',
                                 'type', 'source', 'heat', 'date_time',
                                 'i_keywords_label', 'i_class_label', 'i_entities_label'],
                        sep=';',
                        error_bad_lines=False,
                        encoding='utf-8')
    ratings = pd.read_csv(os.path.join(train_path_dir, 'ratings.csv'),
                          usecols=['user_id', 'user_type', 'item_id', 'rate'],
                          sep=';',
                          error_bad_lines=False,
                          encoding='utf-8')
    ctrs = pd.read_csv(os.path.join(train_path_dir, 'ctr.csv'),
                       usecols=['user_id', 'user_type', 'item_id', 'click'],
                       sep=';',
                       error_bad_lines=False,
                       encoding='utf-8')
    return users, items, ratings, ctrs

def add_features(original_df, base_info):
    logger.debug("------------处理点击率数据-------------")
    filter_user = pd.merge(left=original_df, right=base_info['users'], on=['user_id'],
                           how="inner")  # 用户最少行数相对最少，先用用户表拼接信息
    logger.debug(
        '------------------------filter_user:%s------------------------------------------' % len(filter_user))
    filter_item = pd.merge(left=filter_user, right=base_info['items'], on=['item_id'],
                           how='inner')  # 物品数相对适中，拼接物品信息，过滤掉没有交互的物品
    logger.debug(
        '------------------------filter_item:%s------------------------------------------' % len(filter_item))
    filter_click = pd.merge(left=filter_item, right=base_info['ctrs'],
                            on=['user_id', 'user_type', 'item_id'],
                            how='inner')  # 交互信息最多，拼接交互信息放在最后
    logger.debug(
        '------------------------filter_click:%s------------------------------------------' % len(filter_click))
    filter_click['id'] = filter_click['user_id'] + "|" + (filter_click['user_type']).map(str) + "|" + filter_click[
        'item_id']
    #filter_click.drop(columns=gl.DELETE_COLS, inplace=True)
    filter_click.rename(columns={"click": "target"}, inplace=True)  # 将click作为排序target
    logger.debug(
        '------------------------特征列列名%s------------------------------------------' % filter_click.columns.tolist())
    logger.debug(
        '------------------------特征列样例%s------------------------------------------' % filter_click.head(6))
    logger.debug(filter_click.shape[0])
    return


users, items, ratings, ctrs = load_dataset('C:\\Users\\dell\\PycharmProjects\\com.kgdata.nlp.recommeders_new\\try\\demo_data')
combine_item_clicking = pd.merge(ctrs, items[['item_id']], on='item_id', how='inner')
user_list = list(combine_item_clicking['user_id'].unique())  # 评分列表中的用户取唯一值，即哪些用户给了评分
item_list = list(combine_item_clicking['item_id'].unique())  # 评分列表中的商品取唯一值，即哪些商品被评分了
base_info = {'users': users, 'items': items, 'ratings': ratings, 'ctrs': ctrs,
             "combine_item_clicking": combine_item_clicking, 'user_list': user_list,
             'item_list': item_list}



click_df = add_features(original_df=base_info['combine_item_clicking'], base_info=base_info)

print(click_df.head())