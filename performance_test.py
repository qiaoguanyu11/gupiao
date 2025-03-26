from src.data.stock_manager import StockManager
import time
from datetime import datetime
from collections import defaultdict

def print_performance_stats(timing_stats):
    """打印性能统计信息"""
    print("\n性能统计：")
    print("=" * 50)
    
    # 打印整体操作时间
    if '获取股票列表' in timing_stats:
        print(f"\n获取股票列表总耗时：{timing_stats['获取股票列表']:.2f}秒")
    
    if '获取实时数据' in timing_stats:
        print(f"获取实时数据总耗时：{timing_stats['获取实时数据']:.2f}秒")
    
    # 打印每支股票的处理时间
    print("\n各股票处理时间统计：")
    print("-" * 50)
    
    total_hist_time = 0
    total_calc_time = 0
    stock_count = 0
    
    for code, stats in timing_stats.items():
        if isinstance(stats, dict) and '股票名称' in stats:
            stock_count += 1
            hist_time = stats.get('获取历史数据', 0)
            calc_time = stats.get('计算分析', 0)
            total_hist_time += hist_time
            total_calc_time += calc_time
            
            print(f"\n股票 {code} ({stats['股票名称']}):")
            print(f"  获取历史数据: {hist_time:.2f}秒")
            print(f"  计算分析: {calc_time:.2f}秒")
            print(f"  总处理时间: {hist_time + calc_time:.2f}秒")
    
    if stock_count > 0:
        print("\n平均处理时间：")
        print(f"  历史数据获取平均时间: {total_hist_time/stock_count:.2f}秒")
        print(f"  计算分析平均时间: {total_calc_time/stock_count:.2f}秒")
        print(f"  每股票平均总处理时间: {(total_hist_time + total_calc_time)/stock_count:.2f}秒")

def print_analysis_results(stocks):
    """打印分析结果"""
    print("\n分析结果：")
    print("=" * 50)
    
    for stock in stocks:
        print(f"\n{stock['code']} {stock['name']} "
              f"现价：{stock.get('current_price', 'N/A')} "
              f"成本：{stock['actual_cost']:.2f} "
              f"数量：{stock['quantity']} "
              f"总成本：{stock['total_cost']:.2f} "
              f"入手日期：{stock['register_date'].strftime('%Y-%m-%d')}")
        
        if 'trading_analysis' in stock:
            analysis = stock['trading_analysis']
            target_prices = analysis['target_prices']
            opportunities = analysis['first_opportunities']
            
            # 打印各个目标价位的情况
            for rate in ['3%', '5%', '8%', '10%']:
                target_price = target_prices[rate]
                first_date = opportunities[rate]
                status = f"有机会，首次出现在{first_date}" if first_date else "暂无机会"
                print(f"{rate}收益目标（{target_price:.2f}）：{status}")
            
            # 打印止损情况
            stop_loss = analysis['first_stop_loss']
            stop_loss_price = analysis['stop_loss_price']
            stop_loss_status = f"需要止损，首次出现在{stop_loss}" if stop_loss else "暂无需要止损"
            print(f"止损价（{stop_loss_price:.2f}）：{stop_loss_status}")

def main():
    """主函数"""
    # 测试数据
    test_stocks = [
        "000063", "000422", "000541", "000651", "000678", "000737", "000818", "000977",
        "002049", "002195", "002204", "002261", "002276", "002553", "002594", "002611",
        "002681", "300008", "300059", "300065", "300100", "300115", "300153", "300274",
        "300353", "300476", "300502", "300750", "300766", "300846", "600126", "600320",
        "600363", "600418", "600577", "600580", "600589", "600797", "601608", "601899",
        "603300", "603881", "603986", "603993", "688041", "688256", "688981"
    ]
    
    print(f"\n开始性能测试，测试{len(test_stocks)}个股票...")
    
    # 添加测试数据
    start_time = time.time()
    for code in test_stocks:
        StockManager.add_stock(code, 100.0, 100, "性能测试用")
    add_time = time.time() - start_time
    print(f"\n添加测试数据耗时：{add_time:.2f}秒")
    
    try:
        print("\n开始并行处理...")
        
        # 获取并处理股票数据
        start_time = time.time()
        stocks = StockManager.list_all_stocks()
        process_time = time.time() - start_time
        
        # 提取第一个股票中的性能统计信息
        if stocks and stocks[0].get('timing_stats'):
            print_performance_stats(stocks[0]['timing_stats'])
        
        print(f"\n总处理耗时：{process_time:.2f}秒")
        
        # 打印分析结果
        print_analysis_results(stocks)
        
    finally:
        # 清理测试数据
        print("\n清理测试数据...")
        for code in test_stocks:
            StockManager.remove_stock(code)
        print("测试完成！")

if __name__ == "__main__":
    main() 