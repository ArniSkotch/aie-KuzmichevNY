"""Разовая настройка проекта: создаёт шаблон configs/.env.example (если его нет)
и следит, чтобы configs/.env не попадал в git.

Запуск:
    python -m src.setup_project
"""
from __future__ import annotations

import pathlib

from src.config import PROJECT_ROOT

ENV_EXAMPLE_CONTENT = """# Скопируйте этот файл в configs/.env и заполните значения.
# НИКОГДА не коммитьте configs/.env в репозиторий!

# Kaggle API (https://www.kaggle.com/settings -> API -> Create New Token)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
"""


def ensure_env_example(project_root: pathlib.Path = PROJECT_ROOT) -> pathlib.Path:
    configs_dir = project_root / "configs"
    configs_dir.mkdir(exist_ok=True)

    env_example_path = configs_dir / ".env.example"
    if not env_example_path.exists():
        env_example_path.write_text(ENV_EXAMPLE_CONTENT, encoding="utf-8")
        print(f"Создан шаблон: {env_example_path}")
    else:
        print(f"Шаблон уже существует: {env_example_path}")
    return env_example_path


def ensure_gitignore_entry(project_root: pathlib.Path = PROJECT_ROOT) -> None:
    gitignore_path = project_root / ".gitignore"
    lines = gitignore_path.read_text(encoding="utf-8").splitlines() if gitignore_path.exists() else []
    if "configs/.env" not in lines:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            f.write("\nconfigs/.env\n")
        print("configs/.env добавлен в .gitignore")
    else:
        print("configs/.env уже исключён в .gitignore")


if __name__ == "__main__":
    ensure_env_example()
    ensure_gitignore_entry()
