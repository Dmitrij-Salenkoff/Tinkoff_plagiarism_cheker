import ast
import os
from Levenshtein import distance
import pandas as pd
import glob
from itertools import product, combinations, combinations_with_replacement


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
        'direct': [get_direct_class(f1_path, f2_path)],
        'indirect': [get_indirect_class(f1_path, f2_path)]
    }
    return pd.DataFrame(data)


def to_df_classes():
    file_names = [os.path.split(i)[1] for i in glob.glob("files/*.py")][:50]
    lst = ['files', 'plagiat1', 'plagiat2']
    df_1 = pd.DataFrame()
    for i, j in combinations_with_replacement(lst, 2):
        for n, m in combinations_with_replacement(file_names, 2):
            f1_path = os.path.join(i, n)
            f2_path = os.path.join(j, m)
            df_1 = pd.concat([df_1, get_classes(f1_path, f2_path)], axis=0)
    df_1.to_csv('test_dataset.csv', index=False)


def lev_dist(f1_path: str, f2_path: str):
    with open(f1_path, 'r', encoding="UTF-8") as f1, open(f2_path, 'r', encoding="UTF-8") as f2:
        return distance(f1.read(), f2.read())


if __name__ == '__main__':
    import pylint
    import os
    f_path = 'files/catboost.py'
    #
    # print(pylint.run_pylint([f_path]))

    from pylint.lint import Run

    results = Run([f_path], do_exit=False)
    print(results.linter.stats.error)
    print(results.linter.stats.fatal)
    print(results.linter.stats.global_note)

    # os.system(f'pylint {f_path}')


    # # to_df_classes()
    # df = pd.read_csv('test_dataset.csv')
    # df['test'] = df.apply(lambda x: lev_dist(x['f1_path'], x['f2_path']), axis=1)
    # print(df)

from Levenshtein import distance

