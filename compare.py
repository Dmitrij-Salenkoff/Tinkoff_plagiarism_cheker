import argparse
import pickle
import re
from Levenshtein import distance


def lev_dist(f1_path: str, f2_path: str):
    with open(f1_path, 'r', encoding="UTF-8") as f1, open(f2_path, 'r', encoding="UTF-8") as f2:
        return distance(f1.read(), f2.read())


def lev_dist_norm(f1_path: str, f2_path: str):
    with open(f1_path, 'r', encoding="UTF-8") as f1, open(f2_path, 'r', encoding="UTF-8") as f2:
        str_1 = f1.read()
        str_2 = f2.read()
        return (distance(str_1, str_2) / (len(str_1) + len(str_2) + 1), len(str_1), len(str_2) + 1)


def not_unicode(f1_path):
    with open(f1_path, 'r', encoding="UTF-8") as f:
        return len(re.sub('[\w\s.():",:;\[\]=+\-*`<>^/#{}\'\\\@!?%—~–&|$]', '', f.read()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML model to predict plagiarsm')
    parser.add_argument('input', type=str, help='Folder with general scripts')
    parser.add_argument('scores', type=str, help='Folder with first plagiats')
    parser.add_argument(
        '--model',
        type=str,
        default='model.pkl',
        help='Path to download model'
    )
    args = parser.parse_args()

    pickled_model = pickle.load(open(args.model, 'rb'))

    scores = []

    with open(args.input, 'r') as f:
        for lst_el in f.read().splitlines():
            f1_path, f2_path = lst_el.split(' ')
            features = [lev_dist(f1_path, f2_path), lev_dist_norm(f1_path, f2_path), not_unicode(f1_path),
                        not_unicode(f2_path)]
            scores.append(str(pickled_model.predict_proba(features)))

    with open(args.scores, 'w+') as f:
        f.write('\n'.join(scores))
