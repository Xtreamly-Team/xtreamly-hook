import os
import pandas as pd
from sqlalchemy import create_engine


class Positions:
    def __init__(self):
        db_user = os.environ.get('DB_USERNAME', 'postgres')
        db_pwd = os.environ['DB_PASSWORD']
        db_host = os.environ['DB_HOST']
        db_port = os.environ.get('DB_PORT', 5432)
        db_name = os.environ.get('DB_NAME', 'xtr_trade_db')
        self.engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}')

    def get_active_positions(self, user_id: str):
        sql = 'SELECT * FROM positions WHERE "userId" = %(userId)s AND "status" = %(status)s'
        params = {
            "userId": user_id,
            "status": 'active'
        }
        return pd.read_sql(sql, self.engine, params=params)
