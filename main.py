from src.data.stock_manager import StockManager

def main():
    print("股票数据管理系统启动...")
    
    while True:
        print("\n请选择操作：")
        print("1. 添加关注股票")
        print("2. 查看所有股票")
        print("3. 删除关注股票")
        print("4. 测试数据源")
        print("0. 退出")
        
        choice = input("\n请输入选项（0-4）：")
        
        if choice == "1":
            code = input("请输入股票代码（6位）：")
            try:
                entry_price = float(input("请输入入手价格："))
                quantity = int(input("请输入入手数量（股）："))
                notes = input("请输入备注（可选）：")
                success, message = StockManager.add_stock(code, entry_price, quantity, notes)
                print(message)
            except ValueError:
                print("输入格式错误，请确保价格为数字，数量为整数")
            except Exception as e:
                print(f"添加失败：{str(e)}")
        
        elif choice == "2":
            try:
                stocks = StockManager.list_all_stocks()
                print("\n当前关注的股票：")
                for stock in stocks:
                    print(f"代码：{stock['code']}, 名称：{stock['name']}, "
                          f"入手价：{stock['entry_price']:.2f}, 实际成本：{stock['actual_cost']:.2f}, "
                          f"入手数量：{stock['quantity']}股, 总成本：{stock['total_cost']:.2f}, "
                          f"现价：{stock['current_price']}, 涨跌幅：{stock['daily_change']}, "
                          f"入手日期：{stock['register_date'].strftime('%Y-%m-%d')}")
                    
                    # 显示出手机会
                    print("    入手后3个交易日内的出手机会：")
                    opportunities = stock['selling_opportunities']
                    
                    for rate in ['3%', '5%', '8%', '10%']:
                        opp = opportunities[rate]
                        if opp['first_opportunity'] is not None:
                            first_opp = opp['first_opportunity']
                            daily_counts = opp['daily_counts']
                            print(f"    - {rate}收益率: 首次出手机会 {first_opp['date']} {first_opp['time']}, "
                                  f"最高价：{first_opp['high']:.2f}, "
                                  f"实际到手价：{first_opp['actual_return']:.2f}, "
                                  f"收益率：{first_opp['return_rate']:.2f}%, "
                                  f"实际到手金额：{first_opp['return_amount']:.2f}, "
                                  f"盈利金额：{first_opp['profit_amount']:.2f}")
                            print(f"      后续出手机会统计：第一天 {daily_counts[0]}次, "
                                  f"第二天 {daily_counts[1]}次, "
                                  f"第三天 {daily_counts[2]}次")
                        else:
                            print(f"    - {rate}收益率：3个交易日内未出现出手机会")
                    
                    # 显示止损信息
                    if stock['stop_loss_point']:
                        sl = stock['stop_loss_point']
                        print(f"    ⚠️ 触发止损：{sl['date']} {sl['time']}, "
                              f"最低价：{sl['low_price']:.2f}, "
                              f"实际到手价：{sl['actual_price']:.2f}, "
                              f"亏损率：{sl['loss_rate']:.2f}%, "
                              f"实际到手金额：{sl['return_amount']:.2f}, "
                              f"亏损金额：{sl['loss_amount']:.2f}")
                    else:
                        print("    入手后3个交易日内未触发止损")
                    
                    print()  # 添加空行分隔不同股票
            except Exception as e:
                print(f"获取失败：{str(e)}")
        
        elif choice == "3":
            code = input("请输入要删除的股票代码（6位）：")
            try:
                success, message = StockManager.remove_stock(code)
                print(message)
            except Exception as e:
                print(f"删除失败：{str(e)}")
                
        elif choice == "4":
            code = input("请输入要测试的股票代码（6位）：")
            print("\n=== 测试数据源 ===")
            print("1. 测试分时数据")
            print("2. 测试日线数据")
            print("3. 全部测试")
            test_choice = input("请选择测试类型（1-3）：")
            
            if test_choice in ['1', '3']:
                StockManager.test_minute_data_availability(code)
            if test_choice in ['2', '3']:
                StockManager.test_daily_data_availability(code)
        
        elif choice == "0":
            print("感谢使用，再见！")
            break
        
        else:
            print("无效的选项，请重新选择")

if __name__ == "__main__":
    main() 