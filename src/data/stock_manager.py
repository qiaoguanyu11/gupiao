import json
import os
from datetime import datetime, timedelta
import time
import akshare as ak
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import threading
from collections import defaultdict
import aiohttp
import asyncio
import requests
import mysql.connector

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
    DATABASE_CONFIG = {
        'host': 'localhost',
        'user': 'root',
        'password': 'hepzibah1',
        'database': 'stock_tracker'
    }

    # 更新时间间隔（分钟）
    UPDATE_INTERVAL = 10

    @staticmethod
    def should_update_stock_info(last_update_time):
        """检查是否需要更新股票信息
        Args:
            last_update_time: 上次更新时间
        Returns:
            bool: 是否需要更新
        """
        if not last_update_time:
            return True
        
        current_time = datetime.now()
        time_diff = current_time - last_update_time
        return time_diff.total_seconds() >= StockManager.UPDATE_INTERVAL * 60

    @staticmethod
    def get_db_connection():
        """获取数据库连接"""
        return mysql.connector.connect(**StockManager.DATABASE_CONFIG)

    @staticmethod
    def init_database():
        """初始化数据库表"""
        conn = None
        cursor = None
        try:
            conn = StockManager.get_db_connection()
            cursor = conn.cursor()

            # 创建股票表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    code VARCHAR(10) PRIMARY KEY,
                    cost DECIMAL(10, 2) NOT NULL,
                    quantity INT NOT NULL,
                    total_cost DECIMAL(10, 2) NOT NULL,
                    register_date DATETIME NOT NULL
                )
            """)

            # 创建股票信息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_info (
                    code VARCHAR(10) PRIMARY KEY,
                    name VARCHAR(50) NOT NULL,
                    current_price DECIMAL(10, 2) NOT NULL,
                    update_time DATETIME NOT NULL,
                    FOREIGN KEY (code) REFERENCES stocks(code)
                )
            """)

            # 创建删除历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS delete_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    code VARCHAR(10) NOT NULL,
                    name VARCHAR(50) NOT NULL,
                    price DECIMAL(10, 2) NOT NULL,
                    cost DECIMAL(10, 2) NOT NULL,
                    quantity INT NOT NULL,
                    total_cost DECIMAL(10, 2) NOT NULL,
                    delete_time DATETIME NOT NULL,
                    INDEX idx_code (code),
                    INDEX idx_delete_time (delete_time)
                )
            """)

            conn.commit()
            print("数据库表初始化成功")
            
        except Exception as e:
            print(f"初始化数据库失败：{e}")
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    # 创建缓存实例
    _cache = StockDataCache()
    
    # 批量数据获取的大小
    BATCH_SIZE = 50
    
    @staticmethod
    def get_stock_info(stock_code: str) -> tuple:
        """获取股票信息"""
        try:
            # 获取实时行情
            df = ak.stock_zh_a_spot_em()
            stock_data = df[df['代码'] == stock_code]
            
            if stock_data.empty:
                return False, f"未找到股票{stock_code}的信息"
                
            stock_data = stock_data.iloc[0]
            return True, stock_data['名称']
            
        except Exception as e:
            print(f"获取股票{stock_code}信息失败：{str(e)}")
            return False, str(e)

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
            
            conn = cls.get_db_connection()
            cursor = conn.cursor()
            try:
                # 检查是否已存在
                cursor.execute("SELECT * FROM stocks WHERE code = %s", (code,))
                if cursor.fetchone():
                    return False, "股票已存在"
                
                # 计算实际成本（入手价格 * 100.04%）
                actual_cost = entry_price * 1.0004
                # 计算总实际成本（实际成本 * 股票数量）
                total_cost = actual_cost * quantity
                
                # 添加新股票
                cursor.execute("""
                    INSERT INTO stocks (code, cost, quantity, total_cost, register_date)
                    VALUES (%s, %s, %s, %s, %s)
                """, (code, actual_cost, quantity, total_cost, datetime.now()))
                conn.commit()

                # 添加股票信息
                cursor.execute("""
                    INSERT INTO stock_info (code, name, current_price, update_time)
                    VALUES (%s, %s, %s, %s)
                """, (code, name, float(cls.get_stock_data(code)['current_price']), datetime.now()))
                conn.commit()

                return True, "添加成功"
            finally:
                cursor.close()
            
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
            conn = cls.get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM stocks WHERE code = %s", (code,))
                cursor.execute("DELETE FROM stock_info WHERE code = %s", (code,))
                conn.commit()
                return True, "删除成功"
            finally:
                cursor.close()
        except Exception as e:
            return False, f"删除失败：{str(e)}"
    
    @classmethod
    async def _async_get_stocks_from_db(cls):
        """从数据库获取所有股票信息"""
        conn = cls.get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM stocks")
            stocks = cursor.fetchall()
            
            # 将 Row 对象转换为字典列表
            stock_dicts = []
            for stock in stocks:
                stock_dict = {
                    'code': stock['code'],
                    'name': stock['name'],
                    'notes': stock['notes'],
                    'register_date': stock['register_date'],
                    'entry_price': stock['cost'],
                    'actual_cost': stock['cost'],
                    'quantity': stock['quantity'],
                    'total_cost': stock['total_cost']
                }
                stock_dicts.append(stock_dict)
            return stock_dicts
        finally:
            cursor.close()
            conn.close()

    @classmethod
    async def _async_get_real_time_data(cls, codes):
        """一次性获取所有股票的实时数据"""
        try:
            conn = cls.get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            try:
                # 检查是否需要更新
                cursor.execute("SELECT code, update_time FROM stock_info WHERE code IN (%s)" % ','.join(['%s'] * len(codes)), codes)
                update_times = {row['code']: row['update_time'] for row in cursor.fetchall()}
                
                codes_to_update = [code for code in codes if code not in update_times or cls.should_update_stock_info(update_times[code])]
                
                if not codes_to_update:
                    # 如果所有数据都是最新的，直接从数据库获取
                    cursor.execute("""
                        SELECT code, current_price, name
                        FROM stock_info
                        WHERE code IN (%s)
                    """ % ','.join(['%s'] * len(codes)), codes)
                    
                    result = {}
                    for row in cursor.fetchall():
                        result[row['code']] = {
                            "current_price": float(row['current_price']),
                            "name": row['name']
                        }
                    return result
                
                # 使用akshare获取需要更新的股票数据
                df = await asyncio.get_event_loop().run_in_executor(
                    None, ak.stock_zh_a_spot_em
                )
                
                # 将数据转换为字典格式，加快查找速度
                data_dict = {row['代码']: row for _, row in df.iterrows()}
                
                # 更新数据库并返回结果
                result = {}
                for code in codes:
                    if code in data_dict:
                        stock_data = data_dict[code]
                        current_price = float(stock_data['最新价'])
                        name = stock_data['名称']
                        
                        # 更新数据库
                        cursor.execute("""
                            INSERT INTO stock_info (code, name, current_price, update_time)
                            VALUES (%s, %s, %s, NOW())
                            ON DUPLICATE KEY UPDATE
                                name = VALUES(name),
                                current_price = VALUES(current_price),
                                update_time = NOW()
                        """, (code, name, current_price))
                        
                        result[code] = {
                            "current_price": current_price,
                            "name": name
                        }
                
                conn.commit()
                return result
                
            finally:
                cursor.close()
                
        except Exception as e:
            print(f"获取实时数据失败：{str(e)}")
            return {}
        finally:
            if conn:
                conn.close()

    @classmethod
    async def _async_process_single_stock(cls, stock, real_time_data):
        """异步处理单个股票数据"""
        try:
            # 更新实时数据
            if real_time_data:
                stock.update(real_time_data)
            
            # 计算各个目标价位
            target_rates = {
                '3%': stock['actual_cost'] * 1.03,
                '5%': stock['actual_cost'] * 1.05,
                '8%': stock['actual_cost'] * 1.08,
                '10%': stock['actual_cost'] * 1.10
            }
            stop_loss_price = stock['actual_cost'] * 0.95
            
            # 获取日线数据
            daily_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ak.stock_zh_a_hist(
                    symbol=stock['code'],
                    period="daily",
                    start_date=(stock['register_date'] + timedelta(days=1)).strftime('%Y%m%d'),
                    end_date=(datetime.now()).strftime('%Y%m%d'),
                    adjust="qfq"
                )
            )
            
            # 初始化机会记录
            first_opportunities = {
                '3%': None,
                '5%': None,
                '8%': None,
                '10%': None
            }
            first_stop_loss = None
            
            if daily_data is not None and not daily_data.empty:
                # 分析每个交易日的情况
                trading_day_count = 0
                
                for _, row in daily_data.iterrows():
                    if trading_day_count >= 3:  # 只看前3个交易日
                        break
                    
                    current_date = row['日期']
                    actual_high = float(row['最高'])
                    actual_low = float(row['最低'])
                    
                    # 检查每个目标收益率
                    for rate in ['3%', '5%', '8%', '10%']:
                        if first_opportunities[rate] is None and actual_high >= target_rates[rate]:
                            first_opportunities[rate] = current_date
                    
                    # 检查是否触发止损
                    if first_stop_loss is None and actual_low <= stop_loss_price:
                        first_stop_loss = current_date
                    
                    trading_day_count += 1
            
            # 添加分析结果到股票数据中
            stock['trading_analysis'] = {
                'first_opportunities': first_opportunities,
                'first_stop_loss': first_stop_loss,
                'target_prices': target_rates,
                'stop_loss_price': stop_loss_price
            }
            
            return stock
            
        except Exception as e:
            print(f"处理股票 {stock['code']} 数据时出错：{str(e)}")
            # 确保即使出错也返回基本的分析结果
            stock['trading_analysis'] = {
                'first_opportunities': {rate: None for rate in ['3%', '5%', '8%', '10%']},
                'first_stop_loss': None,
                'target_prices': {
                    '3%': stock['actual_cost'] * 1.03,
                    '5%': stock['actual_cost'] * 1.05,
                    '8%': stock['actual_cost'] * 1.08,
                    '10%': stock['actual_cost'] * 1.10
                },
                'stop_loss_price': stock['actual_cost'] * 0.95
            }
            return stock

    @classmethod
    async def process_stocks(cls):
        """处理所有股票数据的主函数"""
        try:
            # 1. 从数据库获取所有股票信息
            stocks = await cls._async_get_stocks_from_db()
            if not stocks:
                return []

            # 2. 一次性获取所有股票的实时数据
            real_time_data = await cls._async_get_real_time_data([s['code'] for s in stocks])

            # 3. 并行处理每只股票的分析
            tasks = []
            for stock in stocks:
                stock_real_time = real_time_data.get(stock['code'], {})
                tasks.append(cls._async_process_single_stock(stock, stock_real_time))

            # 4. 等待所有分析完成
            results = await asyncio.gather(*tasks)
            
            # 5. 确保每个结果都包含完整的分析信息
            for result in results:
                if 'trading_analysis' not in result:
                    result['trading_analysis'] = {
                        'target_prices': {
                            '3%': result['actual_cost'] * 1.03,
                            '5%': result['actual_cost'] * 1.05,
                            '8%': result['actual_cost'] * 1.08,
                            '10%': result['actual_cost'] * 1.10
                        },
                        'stop_loss_price': result['actual_cost'] * 0.95,
                        'first_opportunities': {
                            '3%': None,
                            '5%': None,
                            '8%': None,
                            '10%': None
                        },
                        'first_stop_loss': None
                    }
            
            return results

        except Exception as e:
            print(f"处理股票数据时出错：{str(e)}")
            return []

    @staticmethod
    def list_all_stocks():
        """获取所有股票信息"""
        conn = None
        cursor = None
        try:
            conn = StockManager.get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT s.code, s.cost, s.quantity, s.total_cost, s.register_date,
                       i.name, i.current_price
                FROM stocks s
                JOIN stock_info i ON s.code = i.code
                ORDER BY s.code
            """)
            
            stocks = cursor.fetchall()
            # 转换日期格式
            for stock in stocks:
                stock['register_date'] = stock['register_date']
                
            return stocks
            
        except Exception as e:
            print(f"获取股票列表失败：{e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
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
            conn = cls.get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    UPDATE stocks SET register_date = %s WHERE code = %s
                """, (new_date, code))
                conn.commit()
                return True, "更新成功"
            finally:
                cursor.close()
        except Exception as e:
            return False, f"更新失败：{str(e)}"

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
            conn = cls.get_db_connection()
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM stocks")
                db_stocks = cursor.fetchall()
                
                for stock in db_stocks:
                    stock_dict = {
                        'code': stock['code'],
                        'name': stock['name'],
                        'notes': stock['notes'],
                        'register_date': stock['register_date'],
                        'entry_price': stock['cost'],
                        'actual_cost': stock['cost'],
                        'quantity': stock['quantity'],
                        'total_cost': stock['total_cost']
                    }
                    # 串行处理每个股票
                    processed_stock = cls._async_process_single_stock(stock_dict, None)
                    stocks_serial.append(processed_stock)
            finally:
                cursor.close()
                conn.close()
            
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

    @staticmethod
    def print_performance_stats(timing_stats):
        """打印性能统计"""
        print("\n整体性能统计：")
        print("-" * 50)
        
        # 打印获取股票列表时间
        if '获取股票列表' in timing_stats:
            elapsed = timing_stats['获取股票列表']
            print("获取股票列表:")
            print(f"  总耗时: {elapsed:.2f}秒")
        print("-" * 50)
        
        # 打印获取实时数据时间
        if '获取实时数据' in timing_stats:
            elapsed = timing_stats['获取实时数据']
            print("获取实时数据:")
            print(f"  总耗时: {elapsed:.2f}秒")
        print("-" * 50)
        
        # 打印每支股票的处理时间
        print("\n每支股票的处理时间统计：")
        print("-" * 50)
        for code, stats in timing_stats.items():
            if isinstance(stats, dict) and '股票名称' in stats:
                print(f"\n股票 {code} ({stats['股票名称']}):")
                print(f"  获取历史数据: {stats.get('获取历史数据', 0.00):.2f}秒")
                print(f"  计算分析: {stats.get('计算分析', 0.00):.2f}秒")
                total_time = (stats.get('获取历史数据', 0) + 
                            stats.get('计算分析', 0))
                print(f"  总处理时间: {total_time:.2f}秒")

    @staticmethod
    def delete_stock(code):
        """删除股票并记录删除历史"""
        try:
            # 获取当前股票信息
            conn = StockManager.get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            # 获取股票信息
            cursor.execute("""
                SELECT s.*, i.name, i.current_price 
                FROM stocks s
                JOIN stock_info i ON s.code = i.code
                WHERE s.code = %s
            """, (code,))
            stock = cursor.fetchone()
            
            if not stock:
                return False, "股票不存在"
            
            # 记录删除历史
            cursor.execute("""
                INSERT INTO delete_history 
                (code, name, price, cost, quantity, total_cost, delete_time)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (
                stock['code'],
                stock['name'],
                stock['current_price'],
                stock['cost'],
                stock['quantity'],
                stock['total_cost']
            ))
            
            # 删除股票
            cursor.execute("DELETE FROM stocks WHERE code = %s", (code,))
            conn.commit()
            
            return True, "删除成功"
            
        except Exception as e:
            conn.rollback()
            return False, str(e)
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def get_delete_history():
        """获取删除历史记录"""
        conn = None
        cursor = None
        try:
            conn = StockManager.get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT * FROM delete_history
                ORDER BY delete_time DESC
            """)
            
            return cursor.fetchall()
            
        except Exception as e:
            print(f"获取删除历史记录失败：{e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close() 