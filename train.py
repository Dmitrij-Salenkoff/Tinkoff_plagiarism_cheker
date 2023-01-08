import argparse
import os
from Levenshtein import distance
import pandas as pd
import glob
from itertools import combinations_with_replacement
import re
import pickle

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split


def get_gen_class(f1_path: str, f2_path: str) -> int:
    dir1, file_name1 = os.path.split(f1_path)
    dir2, file_name2 = os.path.split(f2_path)
    return 1 if file_name1 == file_name2 else 0


def get_direct_class(f1_path: str, f2_path: str) -> int:
    dir1, file_name1 = os.path.split(f1_path)
    dir2, file_name2 = os.path.split(f2_path)
    if get_gen_class(f1_path, f2_path):
        return 1 if (dir1 == 'files') is not (dir2 == 'files') else 0
    else:
        return 0


def is_plagiat(path):
    return ('plagiat' in path)


def get_indirect_class(f1_path: str, f2_path: str) -> int:
    dir1, file_name1 = os.path.split(f1_path)
    dir2, file_name2 = os.path.split(f2_path)
    if get_gen_class(f1_path, f2_path) and f1_path != f2_path:
        return 1 if is_plagiat(f1_path) and is_plagiat(f2_path) else 0
    else:
        return 0


def get_features(f1_path: str, f2_path: str):
    pass


def get_classes(f1_path: str, f2_path: str) -> pd.DataFrame:
    data = {
        'f1_path': [f1_path],
        'f2_path': [f2_path],
        'gen': [get_gen_class(f1_path, f2_path)],
        # 'direct': [get_direct_class(f1_path, f2_path)],
        # 'indirect': [get_indirect_class(f1_path, f2_path)]
    }
    return pd.DataFrame(data)


def to_df_classes(lst_dirs: list[str], to_save: bool = False):
    file_names = [os.path.split(i)[1] for i in glob.glob(f"{lst_dirs[0]}/*.py")][:50]
    lst = lst_dirs
    df_1 = pd.DataFrame()
    for i, j in combinations_with_replacement(lst, 2):
        for n, m in combinations_with_replacement(file_names, 2):
            f1_path = os.path.join(i, n)
            f2_path = os.path.join(j, m)
            df_1 = pd.concat([df_1, get_classes(f1_path, f2_path)], axis=0)
    if to_save:
        df_1.to_csv('test_dataset.csv', index=False)
    return df_1.reset_index().drop(columns='index')


def script_len(f_path: str):
    with open(f_path, 'r', encoding="UTF-8") as f:
        return len(f.read())


def lev_dist(f1_path: str, f2_path: str):
    with open(f1_path, 'r', encoding="UTF-8") as f1, open(f2_path, 'r', encoding="UTF-8") as f2:
        return distance(f1.read(), f2.read())


def clean_from_doc(st: str):
    return re.sub(r'""".*"""', '', st)


def lev_dist_norm(f1_path: str, f2_path: str):
    with open(f1_path, 'r', encoding="UTF-8") as f1, open(f2_path, 'r', encoding="UTF-8") as f2:
        str_1 = f1.read()
        str_2 = f2.read()
        return (distance(str_1, str_2) / (len(str_1) + len(str_2) + 1), len(str_1), len(str_2) + 1)


def lev_dist_norm_clear(f1_path: str, f2_path: str):
    with open(f1_path, 'r', encoding="UTF-8") as f1, open(f2_path, 'r', encoding="UTF-8") as f2:
        str_1 = clean_from_doc(f1.read())
        str_2 = clean_from_doc(f2.read())
        return distance(str_1, str_2) / (len(str_1) + len(str_2) + 1)


def not_unicode(f1_path):
    with open(f1_path, 'r', encoding="UTF-8") as f:
        return len(re.sub('[\w\s.():",:;\[\]=+\-*`<>^/#{}\'\\\@!?%—~–&|$]', '', f.read()))


def get_files_len(df):
    df_buff = pd.DataFrame()
    file_names = list(df['f1_path'].unique())
    for file in file_names:
        with open(file, 'r', encoding="UTF-8") as f:
            df_sec = pd.DataFrame.from_dict({'f_path': [file], 'script_len': [len(str(f.read()))]})
            df_buff = pd.concat([df_buff, df_sec])
    return df_buff


def get_script_len(column_name, df):
    return df.merge(get_files_len(df), left_on=column_name, right_on='f_path')['script_len']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML model to predict plagiarsm')
    parser.add_argument('files', type=str, help='Folder with general scripts')
    parser.add_argument('plagiat1', type=str, help='Folder with first plagiats')
    parser.add_argument('plagiat2', type=str, help='Folder with second plagiats')
    parser.add_argument(
        '--model',
        type=str,
        default='model.pkl',
        help='Path to save model'
    )
    args = parser.parse_args()
    df = to_df_classes([args.files, args.plagiat1, args.plagiat2])
    df['lev'] = df.apply(lambda x: lev_dist(x['f1_path'], x['f2_path']), axis=1)
    df['lev_norm'] = df.apply(lambda x: lev_dist_norm(x['f1_path'], x['f2_path']), axis=1)
    df['not_unicode_1'] = df['f1_path'].apply(not_unicode)
    df['not_unicode_2'] = df['f2_path'].apply(not_unicode)

    model = CatBoostClassifier(iterations=500,
                               learning_rate=0.1,
                               depth=8)
    features = ['lev', 'lev_norm', 'not_unicode_1', 'not_unicode_2']
    target = ['gen']

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

    train_data = Pool(data=X_train, label=y_train)
    test_data = Pool(data=X_test, label=y_test)

    model.fit(train_data, eval_set=test_data)

    pickle.dump(model, open(args.model, 'wb'))
