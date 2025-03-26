from sqlalchemy import create_engine, text
import pymysql

# 数据库配置
DATABASE_URL = "mysql+pymysql://root:hepzibah1@localhost/stock_tracker?charset=utf8mb4"

def test_connection():
    try:
        # 测试直接使用 pymysql 连接
        print("测试 PyMySQL 直接连接...")
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='hepzibah1',
            database='stock_tracker',
            charset='utf8mb4'
        )
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print("数据库表：")
            for table in tables:
                print(f"- {table[0]}")
                
            # 检查 stocks 表中的数据
            cursor.execute("SELECT * FROM stocks")
            stocks = cursor.fetchall()
            print("\n股票数据：")
            for stock in stocks:
                print(f"股票代码：{stock[0]}, 名称：{stock[1]}")
        
        conn.close()
        print("\nPyMySQL 连接测试成功！")
        
    except Exception as e:
        print(f"PyMySQL 连接测试失败：{str(e)}")
    
    try:
        # 测试 SQLAlchemy 连接
        print("\n测试 SQLAlchemy 连接...")
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            result = connection.execute(text("SHOW TABLES"))
            tables = result.fetchall()
            print("数据库表：")
            for table in tables:
                print(f"- {table[0]}")
                
            # 检查 stocks 表中的数据
            result = connection.execute(text("SELECT * FROM stocks"))
            stocks = result.fetchall()
            print("\n股票数据：")
            for stock in stocks:
                print(f"股票代码：{stock[0]}, 名称：{stock[1]}")
                
        print("\nSQLAlchemy 连接测试成功！")
        
    except Exception as e:
        print(f"SQLAlchemy 连接测试失败：{str(e)}")

if __name__ == "__main__":
    test_connection() 