from src.data.stock_manager import StockManager
from datetime import datetime

def main():
    """更新所有股票的入手时间"""
    print("开始更新股票入手时间...")
    
    # 获取所有股票
    results = StockManager.list_all_stocks()
    
    # 设置新的入手时间
    new_date = datetime(2025, 3, 23)
    success_count = 0
    fail_count = 0
    
    for stock in results:
        code = stock['code']
        success, message = StockManager.update_register_date(code, new_date)
        if success:
            success_count += 1
            print(f"成功更新: {code} {stock['name']}")
        else:
            fail_count += 1
            print(f"更新失败: {code} {stock['name']} 原因: {message}")
    
    print(f"\n更新完成！成功: {success_count} 失败: {fail_count}")

if __name__ == "__main__":
    main() 