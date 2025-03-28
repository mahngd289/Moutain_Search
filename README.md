# Install pgvector

- Link tutourial: [ pgvector installation on windows ](https://www.youtube.com/watch?v=YoQZRKjgBkU&ab_channel=KantikAI)

```bash
conda create -n moutain_search_db

conda activate moutain_search_db

conda install -y -c conda-forge postgresql

conda install -c conda-forge pgvector

cd Documents

initdb -D moutain_db

# Start -> Services -> postgresql-x64-16 -> Stop

pg_ctl -D moutain_db -l logfile start

createuser --encrypted --pwprompt non_super_user
# 123456

createdb --owner non_super_user test_db

psql -d test_db

CREATE ROLE admin_user WITH LOGIN SUPERUSER PASSWORD '123456';

CREATE EXTENSION IF NOT EXISTS vector;
```

# Install app environment
```bash
# Pycharm: Add New Interpreters -> Add Local Interpreters -> Name: Mountain_Search -> Type: Conda -> Python version: 3.9

conda activate Mountain_Search

conda install numpy=1.21.6

pip install tensorflow==2.8.0

conda install scikit-learn=1.0.2

conda install opencv=4.5.5

conda install pillow=9.0.1

conda install -c conda-forge psycopg2=2.9.3

pip install protobuf==3.19.6
```