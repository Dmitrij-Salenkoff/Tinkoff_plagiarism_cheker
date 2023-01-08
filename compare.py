"""
- Импорты
- Объявление функций
- Объявление переменных глобально
- Декораторы

Хочу чтобы модель давала скор = 1, если я даю на вход один и тот же файл. А на дубли файлов давала скор > 0.9
На разные файлы (не дубли) давала скор < 0.3

Можно отдельно считать "Прямой плагиат" или "Косвенный"
"""

# with open('files/main.py', 'r') as fd:
#     text = fd.read()
#
#     # body_parse = ast.parse(text).body
#     # astpretty.pprint(body_parse[1])
#
#     for node in ast.iter_child_nodes(ast.parse(text)):
#         if isinstance(node, ast.Import):
#             print(node.names[0])

import ast
from collections import namedtuple


def get_imports(path):
    Import = namedtuple("Import", ["module", "name", "alias"])
    with open(path, 'r', encoding="UTF-8") as fh:
        root = ast.parse(fh.read(), path)

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            module = []
        elif isinstance(node, ast.ImportFrom):
            module = node.module.split('.')
        else:
            continue

        for n in node.names:
            yield Import(module, n.name.split('.'), n.asname)


if __name__ == '__main__':
    for i in get_imports('plagiat1/awac.py'):
        print(i)
