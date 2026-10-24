"""Smoke-тест: playground-скрипты должны как минимум импортироваться.

``playground/`` не входит в ``testpaths`` и не имеет своих тестов, поэтому
поломка импорта там (например, ссылка на переименованный/удалённый модуль)
никак не ловится обычным прогоном ``pytest tests/``. Этот файл — не полный
тест playground-кода, а только гарантия, что модуль не падает при импорте.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_causal_wrapper_imports():
    import playground.causal_wrapper as causal_wrapper

    assert hasattr(causal_wrapper, "CausalRunner")
