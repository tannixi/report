import pandas as pd
from sqlalchemy import create_engine

# 读取 CSV 文件
df = pd.read_csv('C:/Users/12753/Desktop/ground_truths.csv')

# 创建数据库连接引擎
engine = create_engine('mysql+pymysql://user:password@host/dbname')

# 将数据导入 MySQL
df.to_sql('report', con=engine, if_exists='append', index=False)
