import sqlite3
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text

class DBconnection:
    def __init__(self, db_type="sqlite", host=None, user=None, password=None, database="test.db"):
        self.db_type = db_type
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.engine = None

    def get_engine(self):
        if self.db_type == "mysql":
            url = f"mysql+mysqlconnector://{self.user}:{self.password}@{self.host}/{self.database}"
            self.engine = create_engine(url)
            return SQLDatabase(self.engine)
        elif self.db_type == "psql":
            url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}/{self.database}"
            self.engine = create_engine(url)
            return SQLDatabase(self.engine)
        else:
            # SQLite
            url = f"sqlite:///{self.database}"
            self.engine = create_engine(url)
            return SQLDatabase(self.engine)

    def get_sqlalchemy_engine(self):
        """Returns the raw SQLAlchemy engine."""
        if not self.engine:
            self.get_engine() # Initialize engine
        return self.engine
