install:
\tpip install -U pip
\tpip install -e .

dev:
\tpip install black ruff pytest pre-commit
\tpre-commit install

test:
\tpytest -q

lint:
\tblack --check .
\truff check .

fix:
\tblack .
\truff check . --fix
