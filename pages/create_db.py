from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

class DBconnection:
    def __init__(self, db_type="sqlite", host=None, user=None, password=None, database="test.db"):
        self.db_type = db_type
        self.database = database 
        self.connection = [user,password,host,database]
    
    def get_engine(self):
        if self.db_type == "mysql":
            url = f"mysql+mysqlconnector://{self.connection[0]}:{self.connection[1]}@{self.connection[2]}/{self.connection[3]}"
            return SQLDatabase(create_engine(url))
        elif self.db_type == "psql":
            return SQLDatabase(create_engine(f"postgresql+psycopg2://{self.connection[0]}:{self.connection[1]}@{self.connection[2]}/{self.connection[3]}"))
        else:
            return SQLDatabase(create_engine(f"sqlite:///{self.database}"))

    def create_table(self, table_name, columns,dtypes):
        columns = ", ".join([f"{col} {dtype}" for col, dtype in zip(columns.split(','), dtypes.split(','))])
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
        self.connection.commit()     

    def insert_data(self, table_name, data):
        self.cursor.execute(f"INSERT INTO {table_name} VALUES {data}")
        self.connection.commit()
    
    def show_table(self, table_name):
        self.cursor.execute(f"SELECT * FROM {table_name}")
        rows = self.cursor.fetchall()
        for row in rows:
            print(row)

"""# Example usage"""
"""
host = "localhost"
user = "root"
password = "1234"

test = DBconnection("psql",host,user,password,database="test")
test.create_table("test_table", "id,name,age","INT,VARCHAR(25),INT")
test.insert_data("test_table", (1, "John Doe", 30))
test.insert_data("test_table", (2, "Jane Smith", 25))
test.insert_data("test_table", (3, "Alice Johnson", 28))

test.show_table("test_table")
test.close()
"""