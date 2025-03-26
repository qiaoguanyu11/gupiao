import json
import os
from datetime import datetime, timedelta
import time
import akshare as ak
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, DateTime, MetaData
from sqlalchemy.orm import sessionmaker
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import threading
from collections import defaultdict
import requests

class StockDataCache:
    """股票数据缓存类"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.init_cache()
            return cls._instance
    
    def init_cache(self):
        self.real_time_cache = {}
        self.min_data_cache = {}
        self.cache_time = {}
        self.cache_lock = threading.Lock()
    
    def get_cached_data(self, key, cache_type='real_time'):
        """获取缓存数据"""
        cache = self.real_time_cache if cache_type == 'real_time' else self.min_data_cache
        with self.cache_lock:
            if key in cache:
                last_update = self.cache_time.get(key, 0)
                # 实时数据缓存3秒，分时数据缓存5分钟
                max_age = 3 if cache_type == 'real_time' else 300
                if time.time() - last_update < max_age:
                    return cache[key]
        return None
    
    def set_cached_data(self, key, data, cache_type='real_time'):
        """设置缓存数据"""
        cache = self.real_time_cache if cache_type == 'real_time' else self.min_data_cache
        with self.cache_lock:
            cache[key] = data
            self.cache_time[key] = time.time()

class StockManager:
    # 数据库配置
    DATABASE_URL = "mysql+pymysql://root:hepzibah1@localhost/stock_tracker?charset=utf8mb4"
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    
    # 创建缓存实例
    _cache = StockDataCache()
    
    # 批量数据获取的大小
    BATCH_SIZE = 50
    
    # 创建元数据和表
    metadata = MetaData()
    stocks_table = Table(
        'stocks', metadata,
        Column('code', String(10), primary_key=True),
        Column('name', String(50), nullable=False),
        Column('notes', String(200)),
        Column('register_date', DateTime, default=datetime.now),
        Column('entry_price', Float, nullable=False),  # 入手价格
        Column('actual_cost', Float, nullable=False),   # 实际成本
        Column('quantity', Integer, nullable=False),    # 入手数量（股数）
        Column('total_cost', Float, nullable=False)    # 总实际成本
    )
    
    # 确保表存在
    metadata.create_all(engine)
    
    @staticmethod
    def get_stock_info(stock_code: str) -> tuple:
        """获取股票信息"""
        try:
            # 获取实时行情
            df = ak.stock_zh_a_spot_em()
            stock_data = df[df['代码'] == stock_code].iloc[0]
            
            return (
                stock_code,
                stock_data['名称'],
                float(stock_data['最新价']),
                float(stock_data['涨跌幅']),
                int(stock_data['成交量']),
                float(stock_data['成交额'])
            )
        except Exception as e:
            print(f"获取股票{stock_code}信息失败：{str(e)}")
            return None

    @staticmethod
    def get_stock_data(stock_code: str) -> dict:
        """获取股票详细数据"""
        try:
            # 获取实时行情
            stock_df = ak.stock_zh_a_spot_em()
            stock_info = stock_df[stock_df['代码'] == stock_code]
            
            if stock_info.empty:
                return None
                
            # 获取5日成交额
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')  # 多取几天，防止节假日
            hist_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="")
            
            # 计算最近5个交易日的平均成交额
            five_day_amount = hist_data['成交额'].tail(5).mean() / 100000000  # 转换为亿元
            
            return {
                'current_price': float(stock_info.iloc[0]['最新价']),
                'daily_change': float(stock_info.iloc[0]['涨跌幅']),
                'daily_amount': float(stock_info.iloc[0]['成交额']) / 100000000,  # 转换为亿元
                'five_day_amount': round(five_day_amount, 2),
                'turnover_rate': float(stock_info.iloc[0]['换手率'])
            }
        except Exception as e:
            print(f"获取股票数据失败: {str(e)}")
            return None

    @classmethod
    def add_stock(cls, code, entry_price, quantity, notes=""):
        try:
            # 验证股票代码
            if not cls._is_valid_stock_code(code):
                return False, "无效的股票代码"
            
            # 获取股票信息
            success, name = cls.get_stock_info(code)
            if not success:
                return False, name
            
            session = cls.Session()
            try:
                # 检查是否已存在
                if session.query(cls.stocks_table).filter_by(code=code).first():
                    return False, "股票已存在"
                
                # 计算实际成本（入手价格 * 100.04%）
                actual_cost = entry_price * 1.0004
                # 计算总实际成本（实际成本 * 股票数量）
                total_cost = actual_cost * quantity
                
                # 添加新股票
                new_stock = {
                    "code": code,
                    "name": name,
                    "notes": notes,
                    "register_date": datetime.now(),
                    "entry_price": entry_price,
                    "actual_cost": actual_cost,
                    "quantity": quantity,
                    "total_cost": total_cost
                }
                session.execute(cls.stocks_table.insert().values(**new_stock))
                session.commit()
                return True, "添加成功"
            finally:
                session.close()
            
        except Exception as e:
            return False, f"添加失败：{str(e)}"

    @classmethod
    def get_kline_data(cls, code):
        """获取股票K线数据"""
        try:
            # 使用 akshare 获取日K数据
            df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                  start_date=(datetime.now() - timedelta(days=90)).strftime("%Y%m%d"),
                                  end_date=datetime.now().strftime("%Y%m%d"),
                                  adjust="qfq")
            
            # 转换列名以适配 mplfinance
            df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Amount', 'Amplitude', 'PctChg', 'Turnover']
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # 只保留需要的列
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            return df
            
        except Exception as e:
            print(f"获取K线数据失败：{str(e)}")
            return None

    @classmethod
    def remove_stock(cls, code):
        try:
            session = cls.Session()
            try:
                result = session.execute(cls.stocks_table.delete().where(cls.stocks_table.c.code == code))
                session.commit()
                if result.rowcount > 0:
                    return True, "删除成功"
                return False, "股票不存在"
            finally:
                session.close()
        except Exception as e:
            return False, f"删除失败：{str(e)}"
    
    @classmethod
    def _batch_get_real_time_data(cls, codes):
        """批量获取实时数据"""
        start_time = time.time()
        try:
            # 使用 akshare 一次性获取所有股票的实时行情
            df = ak.stock_zh_a_spot_em()
            api_time = time.time() - start_time
            print(f"[性能] 实时行情API调用耗时: {api_time:.2f}秒")
            
            result = {}
            process_start = time.time()
            # 将数据转换为字典格式，加快查找速度
            data_dict = {row['代码']: row for _, row in df.iterrows()}
            
            for code in codes:
                if code in data_dict:
                    stock_data = data_dict[code]
                    current_price = float(stock_data['最新价'])
                    cls._cache.set_cached_data(code, {
                        "current_price": str(current_price),
                        "daily_change": f"{float(stock_data['涨跌幅']):.2f}%"
                    })
                    result[code] = cls._cache.get_cached_data(code)
            
            process_time = time.time() - process_start
            print(f"[性能] 实时数据处理耗时: {process_time:.2f}秒")
            return result
        except Exception as e:
            print(f"批量获取实时数据失败：{str(e)}")
            return {}

    @classmethod
    def _process_stock_batch(cls, stock_batch):
        """处理一批股票数据"""
        # 批量获取实时数据
        codes = [stock['code'] for stock in stock_batch]
        real_time_data = cls._batch_get_real_time_data(codes)
        
        # 使用线程池处理每个股票的出手机会和止损点
        with ThreadPoolExecutor(max_workers=len(stock_batch)) as executor:
            futures = []
            for stock in stock_batch:
                future = executor.submit(cls._process_single_stock, stock, real_time_data.get(stock['code']))
                futures.append(future)
            
            # 收集结果
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"处理股票数据时出错：{str(e)}")
            
            return results

    @classmethod
    def _process_single_stock(cls, stock_dict, real_time_data):
        """处理单个股票数据"""
        start_time = time.time()
        try:
            if real_time_data:
                stock_dict.update(real_time_data)
            
            # 使用缓存获取分时数据
            cache_start = time.time()
            cached_min_data = cls._cache.get_cached_data(stock_dict['code'], 'min_data')
            cache_time = time.time() - cache_start
            
            if cached_min_data:
                opportunities = cached_min_data.get('opportunities')
                stop_loss = cached_min_data.get('stop_loss')
                print(f"[性能] 股票{stock_dict['code']}使用缓存数据，耗时: {cache_time:.2f}秒")
            else:
                calc_start = time.time()
                # 计算出手机会和止损点
                opportunities = cls.check_selling_opportunities(
                    stock_dict['code'],
                    stock_dict['register_date'],
                    stock_dict['actual_cost'],
                    stock_dict['quantity']
                )
                stop_loss = cls.check_stop_loss_point(
                    stock_dict['code'],
                    stock_dict['register_date'],
                    stock_dict['actual_cost'],
                    stock_dict['quantity']
                )
                calc_time = time.time() - calc_start
                print(f"[性能] 股票{stock_dict['code']}计算数据耗时: {calc_time:.2f}秒")
                
                # 缓存计算结果
                cls._cache.set_cached_data(stock_dict['code'], {
                    'opportunities': opportunities,
                    'stop_loss': stop_loss
                }, 'min_data')
            
            stock_dict['selling_opportunities'] = opportunities
            stock_dict['stop_loss_point'] = stop_loss
            
            total_time = time.time() - start_time
            print(f"[性能] 股票{stock_dict['code']}总处理耗时: {total_time:.2f}秒")
            return stock_dict
        except Exception as e:
            print(f"处理股票 {stock_dict['code']} 数据时出错：{str(e)}")
            return stock_dict

    @classmethod
    def list_all_stocks(cls):
        try:
            session = cls.Session()
            try:
                # 使用 select 语句获取所有字段
                query = cls.stocks_table.select()
                stocks = session.execute(query).fetchall()
                
                # 将 Row 对象转换为字典列表
                stock_dicts = []
                for stock in stocks:
                    stock_dict = {
                        'code': stock.code,
                        'name': stock.name,
                        'notes': stock.notes,
                        'register_date': stock.register_date,
                        'entry_price': stock.entry_price,
                        'actual_cost': stock.actual_cost,
                        'quantity': stock.quantity,
                        'total_cost': stock.total_cost
                    }
                    stock_dicts.append(stock_dict)
                
                # 将股票列表分成批次处理
                batches = [stock_dicts[i:i + cls.BATCH_SIZE] 
                          for i in range(0, len(stock_dicts), cls.BATCH_SIZE)]
                
                # 使用进程池处理批次数据
                with Pool(processes=min(cpu_count(), len(batches))) as pool:
                    results = pool.map(cls._process_stock_batch, batches)
                
                # 合并所有结果
                final_results = []
                for batch_result in results:
                    final_results.extend(batch_result)
                
                return final_results
            finally:
                session.close()
        except Exception as e:
            print(f"获取股票列表失败：{str(e)}")
            return []
    
    @classmethod
    def _is_valid_stock_code(cls, code):
        # 简单的股票代码格式验证
        if not code:
            return False
        
        # 支持沪深股票代码格式
        if len(code) != 6:
            return False
        
        return code.isdigit()
    
    @classmethod
    def update_register_date(cls, code, new_date):
        """更新股票的入手日期"""
        try:
            session = cls.Session()
            try:
                result = session.execute(
                    cls.stocks_table.update()
                    .where(cls.stocks_table.c.code == code)
                    .values(register_date=new_date)
                )
                session.commit()
                if result.rowcount > 0:
                    return True, "更新成功"
                return False, "股票不存在"
            finally:
                session.close()
        except Exception as e:
            return False, f"更新失败：{str(e)}"

    @classmethod
    def check_selling_opportunities(cls, stock_code, register_date, actual_cost, quantity):
        """检查股票在入手后3个交易日内的出手机会（T+1交易模式）"""
        # 初始化结果字典
        opportunities = {
            '3%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]},
            '5%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]},
            '8%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]},
            '10%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]}
        }

        # 设置起始日期为注册日期的下一个交易日（T+1）
        current_date = register_date + timedelta(days=1)
        end_datetime = register_date + timedelta(days=5)  # 考虑节假日，多给两天
        trading_days = 0

        while current_date <= end_datetime and trading_days < 3:
            try:
                # 获取分时数据
                date_str = current_date.strftime('%Y%m%d')
                min_data = ak.stock_zh_a_hist_min_em(symbol=stock_code, 
                                                    start_date=date_str, 
                                                    end_date=date_str)
                
                if not min_data.empty:  # 如果有数据，说明是交易日
                    # 检查每个分时点
                    for _, row in min_data.iterrows():
                        high_price = float(row['最高'])
                        # 考虑卖出费用（假设为0.4%）
                        actual_high = high_price * 0.996
                        
                        # 计算实际收益率
                        return_rate = (actual_high / actual_cost - 1)
                        
                        # 检查不同收益率目标
                        if return_rate >= 0.03:  # 3%
                            opportunities['3%']['daily_counts'][trading_days] += 1
                            if opportunities['3%']['first_opportunity'] is None:
                                opportunities['3%']['first_opportunity'] = {
                                    'time': row['时间'],
                                    'price': high_price,
                                    'return_rate': return_rate
                                }
                        
                        if return_rate >= 0.05:  # 5%
                            opportunities['5%']['daily_counts'][trading_days] += 1
                            if opportunities['5%']['first_opportunity'] is None:
                                opportunities['5%']['first_opportunity'] = {
                                    'time': row['时间'],
                                    'price': high_price,
                                    'return_rate': return_rate
                                }
                        
                        if return_rate >= 0.08:  # 8%
                            opportunities['8%']['daily_counts'][trading_days] += 1
                            if opportunities['8%']['first_opportunity'] is None:
                                opportunities['8%']['first_opportunity'] = {
                                    'time': row['时间'],
                                    'price': high_price,
                                    'return_rate': return_rate
                                }
                        
                        if return_rate >= 0.10:  # 10%
                            opportunities['10%']['daily_counts'][trading_days] += 1
                            if opportunities['10%']['first_opportunity'] is None:
                                opportunities['10%']['first_opportunity'] = {
                                    'time': row['时间'],
                                    'price': high_price,
                                    'return_rate': return_rate
                                }
                    
                    trading_days += 1
                
            except Exception as e:
                print(f"获取股票{stock_code}在{date_str}的分时数据时出错：{str(e)}")
            
            current_date += timedelta(days=1)
        
        return opportunities

    @classmethod
    def check_stop_loss_point(cls, stock_code, register_date, actual_cost, quantity):
        """检查股票在入手后3个交易日内的止损点（T+1交易模式）"""
        try:
            # 获取入手日期后的5个自然日数据（为了确保能获取到3个交易日）
            end_date = (register_date + timedelta(days=5)).strftime('%Y%m%d')
            start_date = (register_date + timedelta(days=1)).strftime('%Y%m%d')  # T+1：从下一天开始
            
            stop_loss_price = actual_cost * 0.95  # 止损价格（实际成本的95%）
            total_cost = actual_cost * quantity  # 总实际成本
            
            trading_days = 0
            current_date = datetime.strptime(start_date, '%Y%m%d')
            end_datetime = datetime.strptime(end_date, '%Y%m%d')
            
            # 用于记录最佳止损点
            best_stop_loss = {
                'diff': float('inf'),
                'data': None
            }
            
            while current_date <= end_datetime and trading_days < 3:
                date_str = current_date.strftime('%Y%m%d')
                try:
                    # 获取分时数据
                    min_data = ak.stock_zh_a_hist_min_em(symbol=stock_code, 
                                                        start_date=date_str, 
                                                        end_date=date_str)
                    
                    if not min_data.empty:  # 如果有数据，说明是交易日
                        trading_days += 1
                        
                        # 检查每个分时点
                        for _, row in min_data.iterrows():
                            low_price = float(row['最低'])
                            actual_price = low_price * 0.996  # 考虑卖出费用
                            
                            if actual_price <= stop_loss_price:
                                # 计算与止损价格的差距
                                price_diff = abs(actual_price - stop_loss_price)
                                
                                # 如果这个价格比之前记录的更接近止损价格
                                if price_diff < best_stop_loss['diff']:
                                    best_stop_loss['diff'] = price_diff
                                    actual_return_amount = actual_price * quantity  # 实际到手金额
                                    loss_amount = actual_return_amount - total_cost  # 亏损金额
                                    best_stop_loss['data'] = {
                                        'date': current_date.strftime('%Y-%m-%d'),
                                        'time': row['时间'],
                                        'low_price': low_price,
                                        'actual_price': actual_price,
                                        'loss_rate': (actual_price / actual_cost - 1) * 100,
                                        'return_amount': actual_return_amount,
                                        'loss_amount': loss_amount
                                    }
                except Exception as e:
                    pass
                
                current_date += timedelta(days=1)
            
            return best_stop_loss['data']
            
        except Exception as e:
            return None

    @classmethod
    def test_minute_data_availability(cls, stock_code):
        """测试分时数据的可用性"""
        try:
            print(f"\n测试股票 {stock_code} 的分时数据获取情况：")
            
            # 测试最近7天
            current_date = datetime.now()
            for i in range(7):
                test_date = current_date - timedelta(days=i)
                date_str = test_date.strftime('%Y%m%d')
                print(f"\n检查日期：{date_str}")
                
                try:
                    # 尝试获取分时数据
                    min_data = ak.stock_zh_a_hist_min_em(symbol=stock_code, 
                                                        start_date=date_str, 
                                                        end_date=date_str)
                    
                    if min_data is not None and not min_data.empty:
                        print(f"✓ 成功获取数据，数据点数量：{len(min_data)}")
                        print("数据示例：")
                        print(min_data.head())
                    else:
                        print("✗ 未获取到数据")
                        
                except Exception as e:
                    print(f"✗ 获取失败：{str(e)}")
                    
        except Exception as e:
            print(f"测试过程出错：{str(e)}")
            
    @classmethod
    def test_daily_data_availability(cls, stock_code):
        """测试日线数据的可用性"""
        try:
            print(f"\n测试股票 {stock_code} 的日线数据获取情况：")
            
            # 获取最近7天的日线数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            try:
                # 尝试获取日线数据
                daily_data = ak.stock_zh_a_hist(symbol=stock_code, 
                                              period="daily",
                                              start_date=start_date,
                                              end_date=end_date,
                                              adjust="qfq")
                
                if daily_data is not None and not daily_data.empty:
                    print(f"✓ 成功获取数据，数据点数量：{len(daily_data)}")
                    print("数据示例：")
                    print(daily_data)
                else:
                    print("✗ 未获取到数据")
                    
            except Exception as e:
                print(f"✗ 获取失败：{str(e)}")
                
        except Exception as e:
            print(f"测试过程出错：{str(e)}")

    @classmethod
    def test_performance(cls, stock_codes):
        """测试性能
        Args:
            stock_codes: 要测试的股票代码列表
        """
        print(f"\n开始性能测试，测试{len(stock_codes)}个股票...")
        
        # 先添加测试股票
        for code in stock_codes:
            cls.add_stock(code, 100.0, 100, "性能测试用")
        
        try:
            # 测试并行处理
            print("\n1. 测试并行处理性能...")
            start_time = time.time()
            stocks = cls.list_all_stocks()
            parallel_time = time.time() - start_time
            print(f"并行处理耗时: {parallel_time:.2f}秒")
            print(f"平均每个股票耗时: {parallel_time/len(stock_codes):.2f}秒")
            print(f"CPU核心数: {cpu_count()}")
            
            # 测试串行处理
            print("\n2. 测试串行处理性能...")
            start_time = time.time()
            stocks_serial = []
            session = cls.Session()
            try:
                query = cls.stocks_table.select()
                db_stocks = session.execute(query).fetchall()
                
                for stock in db_stocks:
                    stock_dict = {
                        'code': stock.code,
                        'name': stock.name,
                        'notes': stock.notes,
                        'register_date': stock.register_date,
                        'entry_price': stock.entry_price,
                        'actual_cost': stock.actual_cost,
                        'quantity': stock.quantity,
                        'total_cost': stock.total_cost
                    }
                    # 串行处理每个股票
                    processed_stock = cls._process_single_stock(stock_dict, None)
                    stocks_serial.append(processed_stock)
            finally:
                session.close()
            
            serial_time = time.time() - start_time
            print(f"串行处理耗时: {serial_time:.2f}秒")
            print(f"平均每个股票耗时: {serial_time/len(stock_codes):.2f}秒")
            
            # 计算性能提升
            speedup = serial_time / parallel_time
            print(f"\n性能提升: {speedup:.2f}倍")
            
        finally:
            # 清理测试数据
            print("\n清理测试数据...")
            for code in stock_codes:
                cls.remove_stock(code)
            print("测试完成！")

    def check_opportunities(self, stock_codes):
        """检查多个股票的出手机会"""
        opportunities = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for code in stock_codes:
                # 获取股票信息
                stock_info = self.get_stock_info(code)
                if stock_info:
                    code, name, price, change, volume, amount = stock_info
                    # 模拟检查出手机会
                    register_date = datetime.now() - timedelta(days=1)  # 假设昨天买入
                    actual_cost = price / 1.03  # 假设在当前价格的97%买入
                    quantity = 100  # 假设买入100股
                    
                    future = executor.submit(
                        self.check_selling_opportunities,
                        code, register_date, actual_cost, quantity
                    )
                    futures.append((code, name, price, change, future))
            
            # 收集结果
            for code, name, price, change, future in futures:
                try:
                    result = future.result()
                    if any(level['first_opportunity'] is not None for level in result.values()):
                        opportunities.append({
                            'code': code,
                            'name': name,
                            'current_price': price,
                            'price_change': f"{change}%",
                            'opportunities': result
                        })
                except Exception as e:
                    print(f"处理股票{code}的结果时出错：{str(e)}")
        
        return opportunities 