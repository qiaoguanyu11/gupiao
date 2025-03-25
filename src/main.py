from src.data.stock_manager import StockManager

def print_menu():
    print("\n=== 股票管理系统 ===")
    print("1. 添加股票")
    print("2. 查看所有股票")
    print("3. 删除股票")
    print("4. 退出")
    print("================")

def get_float_price():
    """获取价格输入，支持中文句号"""
    while True:
        try:
            price_str = input("请输入登记价格: ").replace("。", ".")  # 将中文句号替换为英文小数点
            return float(price_str)
        except ValueError:
            print("价格格式不正确，请输入正确的数字（例如：1573.75）")

def main():
    while True:
        print_menu()
        choice = input("请选择操作 (1-4): ")
        
        if choice == "1":
            code = input("请输入股票代码: ")
            price = get_float_price()
            notes = input("请输入备注 (可选): ")
            success, message = StockManager.add_stock(code, price, notes)
            print(message)
            
        elif choice == "2":
            stocks = StockManager.list_all_stocks()
            print("\n当前记录的股票:")
            print("代码\t名称\t登记价\t现价\t日涨跌幅\t累计涨跌幅\t当日成交额\t5日均额\t换手率\t登记日期\t登记时间\t备注")
            print("-" * 150)
            for stock in stocks:
                print(f"{stock['code']}\t{stock['name']}\t{stock['register_price']}\t"
                      f"{stock['current_price']}\t{stock['daily_change']}\t"
                      f"{stock['register_change']}\t{stock['daily_amount']}\t"
                      f"{stock['five_day_amount']}\t{stock['turnover_rate']}\t"
                      f"{stock['register_date']}\t{stock['register_time']}\t{stock['notes']}")
                
        elif choice == "3":
            code = input("请输入要删除的股票代码: ")
            success, message = StockManager.remove_stock(code)
            print(message)
            
        elif choice == "4":
            print("感谢使用！")
            break
            
        else:
            print("无效的选择，请重试")

if __name__ == "__main__":
    main() 