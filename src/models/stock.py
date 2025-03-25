from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class StockRecord(Base):
    __tablename__ = 'stock_records'

    id = Column(Integer, primary_key=True)
    stock_code = Column(String(10), nullable=False)  # 股票代码
    stock_name = Column(String(50))  # 股票名称
    register_price = Column(Float, nullable=False)  # 登记时的价格
    register_date = Column(DateTime, default=datetime.now)  # 登记日期
    register_time = Column(String(10))  # 登记时间（例如：14:30）
    notes = Column(String(200))  # 备注

    def __repr__(self):
        return f"<Stock {self.stock_code} - {self.stock_name}>"

# 创建MySQL数据库连接
DATABASE_URL = "mysql+pymysql://root:hepzibah1@localhost/stock_tracker?charset=utf8mb4"
engine = create_engine(DATABASE_URL)

# 删除旧表（如果存在）
Base.metadata.drop_all(engine)

# 创建新表
Base.metadata.create_all(engine)

# 创建会话
Session = sessionmaker(bind=engine)
session = Session() 