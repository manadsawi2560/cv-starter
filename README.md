# Computer Vision Starter

A lightweight, production-minded template for computer vision projects (clean `src/` layout, tests, pre-commit, CI).

![CI](https://github.com/<USER>/cv-starter/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Quickstart
```bash
# create / activate your env (conda or venv)
pip install -U pip
pip install -e .
pip install black ruff pytest pre-commit
pre-commit install
pytest -q