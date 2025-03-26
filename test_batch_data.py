import akshare as ak
import time
from datetime import datetime, timedelta
import pandas as pd

def batch_get_realtime_data(stock_codes):
    """批量获取实时数据"""
    start_time = time.time()
    try:
        # 使用 akshare 一次性获取所有股票的实时行情
        df = ak.stock_zh_a_spot_em()
        api_time = time.time() - start_time
        print(f"[性能] 实时行情API调用耗时: {api_time:.2f}秒")
        
        # 将数据转换为字典格式，加快查找速度
        data_dict = {row['代码']: row for _, row in df.iterrows()}
        
        result = {}
        process_start = time.time()
        for code in stock_codes:
            if code in data_dict:
                stock_data = data_dict[code]
                result[code] = {
                    "current_price": float(stock_data['最新价']),
                    "daily_change": f"{float(stock_data['涨跌幅']):.2f}%",
                    "volume": float(stock_data['成交量']),
                    "amount": float(stock_data['成交额']),
                    "high": float(stock_data['最高']),
                    "low": float(stock_data['最低'])
                }
        
        process_time = time.time() - process_start
        print(f"[性能] 数据处理耗时: {process_time:.2f}秒")
        return result
    except Exception as e:
        print(f"批量获取实时数据失败：{str(e)}")
        return {}

def batch_get_minute_data(stock_codes, date=None):
    """批量获取分时数据"""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    start_time = time.time()
    result = {}
    
    for code in stock_codes:
        try:
            code_start_time = time.time()
            min_data = ak.stock_zh_a_hist_min_em(symbol=code, 
                                                start_date=date, 
                                                end_date=date)
            
            if not min_data.empty:
                result[code] = min_data
                
            code_time = time.time() - code_start_time
            print(f"[性能] 股票{code}分时数据获取耗时: {code_time:.2f}秒")
            
        except Exception as e:
            print(f"获取股票{code}分时数据失败：{str(e)}")
    
    total_time = time.time() - start_time
    print(f"[性能] 批量获取分时数据总耗时: {total_time:.2f}秒")
    return result

def test_batch_data():
    # 测试用的股票代码列表
    test_stocks = [
        "600519",  # 贵州茅台
        "000858",  # 五粮液
        "601318",  # 中国平安
        "600036",  # 招商银行
        "000333",  # 美的集团
    ]
    
    print("\n1. 测试批量获取实时数据")
    print("-" * 50)
    realtime_data = batch_get_realtime_data(test_stocks)
    for code, data in realtime_data.items():
        print(f"\n股票代码: {code}")
        for key, value in data.items():
            print(f"{key}: {value}")
    
    print("\n2. 测试批量获取分时数据")
    print("-" * 50)
    minute_data = batch_get_minute_data(test_stocks)
    for code, data in minute_data.items():
        print(f"\n股票代码: {code}")
        print(f"数据点数量: {len(data)}")
        print("最新5条数据:")
        print(data.tail())

if __name__ == "__main__":
    test_batch_data() 