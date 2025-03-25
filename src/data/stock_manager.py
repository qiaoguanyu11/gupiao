import json
import os
from datetime import datetime, timedelta
import akshare as ak
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, DateTime, MetaData
from sqlalchemy.orm import sessionmaker

class StockManager:
    # 数据库配置
    DATABASE_URL = "mysql+pymysql://root:hepzibah1@localhost/stock_tracker?charset=utf8mb4"
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    
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
        """获取股票基本信息"""
        try:
            # 获取实时行情
            stock_df = ak.stock_zh_a_spot_em()
            stock_info = stock_df[stock_df['代码'] == stock_code]
            if not stock_info.empty:
                return True, stock_info.iloc[0]['名称']
            return False, "未找到该股票"
        except Exception as e:
            return False, f"获取股票信息失败: {str(e)}"

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
    def list_all_stocks(cls):
        try:
            session = cls.Session()
            try:
                # 使用 select 语句获取所有字段
                query = cls.stocks_table.select()
                stocks = session.execute(query).fetchall()
                result = []
                
                for stock in stocks:
                    # 将 Row 对象转换为字典
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
                    
                    # 获取实时数据
                    real_time_data = cls._get_real_time_data(stock_dict['code'])
                    if real_time_data:
                        stock_dict.update(real_time_data)
                    
                    # 检查出手机会
                    opportunities = cls.check_selling_opportunities(
                        stock_dict['code'],
                        stock_dict['register_date'],
                        stock_dict['actual_cost'],
                        stock_dict['quantity']
                    )
                    stock_dict['selling_opportunities'] = opportunities
                    
                    # 检查止损点
                    stop_loss = cls.check_stop_loss_point(
                        stock_dict['code'],
                        stock_dict['register_date'],
                        stock_dict['actual_cost'],
                        stock_dict['quantity']
                    )
                    stock_dict['stop_loss_point'] = stop_loss
                    
                    result.append(stock_dict)
                
                return result
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
    def _get_real_time_data(cls, code):
        try:
            # 使用 akshare 获取实时行情
            df = ak.stock_zh_a_spot_em()
            stock_data = df[df['代码'] == code].iloc[0]
            current_price = float(stock_data['最新价'])
            
            # 从数据库获取实际成本
            session = cls.Session()
            try:
                stock = session.execute(cls.stocks_table.select().where(cls.stocks_table.c.code == code)).first()
                if stock:
                    actual_cost = stock.actual_cost
                    # 计算基于实际成本的涨跌幅
                    change_rate = ((current_price - actual_cost) / actual_cost) * 100
                    return {
                        "current_price": str(current_price),
                        "daily_change": f"{change_rate:.2f}%"
                    }
            finally:
                session.close()
            
            return {
                "current_price": str(current_price),
                "daily_change": "--"
            }
        except Exception as e:
            print(f"获取实时数据失败：{str(e)}")
            return {
                "current_price": "--",
                "daily_change": "--"
            }

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
        try:
            # 获取入手日期后的5个自然日数据（为了确保能获取到3个交易日）
            end_date = (register_date + timedelta(days=5)).strftime('%Y%m%d')
            start_date = (register_date + timedelta(days=1)).strftime('%Y%m%d')  # T+1：从下一天开始
            
            # 定义不同收益率档位
            target_rates = [0.03, 0.05, 0.08, 0.10]  # 3%, 5%, 8%, 10%
            opportunities = {
                '3%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]},
                '5%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]},
                '8%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]},
                '10%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]}
            }
            
            trading_days = 0
            current_date = datetime.strptime(start_date, '%Y%m%d')
            end_datetime = datetime.strptime(end_date, '%Y%m%d')
            
            while current_date <= end_datetime and trading_days < 3:
                date_str = current_date.strftime('%Y%m%d')
                try:
                    # 获取分时数据
                    min_data = ak.stock_zh_a_hist_min_em(symbol=stock_code, 
                                                        start_date=date_str, 
                                                        end_date=date_str)
                    
                    if not min_data.empty:  # 如果有数据，说明是交易日
                        # 检查每个分时点
                        for _, row in min_data.iterrows():
                            high_price = float(row['最高'])
                            actual_high = high_price * 0.996  # 考虑卖出费用
                            return_rate = (actual_high / actual_cost - 1)
                            actual_return_amount = actual_high * quantity  # 实际到手金额
                            profit_amount = actual_return_amount - (actual_cost * quantity)  # 盈利金额
                            
                            # 检查每个收益率档位
                            for rate in target_rates:
                                rate_key = f"{int(rate * 100)}%"
                                target_price = actual_cost * (1 + rate)  # 目标价格
                                
                                # 如果价格大于等于目标价格
                                if actual_high >= target_price:
                                    # 增加当天的出手机会计数
                                    opportunities[rate_key]['daily_counts'][trading_days] += 1
                                    
                                    # 如果是第一次出现这个收益率的机会，记录详细信息
                                    if opportunities[rate_key]['first_opportunity'] is None:
                                        opportunities[rate_key]['first_opportunity'] = {
                                            'date': current_date.strftime('%Y-%m-%d'),
                                            'time': row['时间'],
                                            'high': high_price,
                                            'actual_return': actual_high,
                                            'return_rate': return_rate * 100,
                                            'return_amount': actual_return_amount,
                                            'profit_amount': profit_amount
                                        }
                        
                        trading_days += 1
                        
                except Exception as e:
                    pass
                
                current_date += timedelta(days=1)
            
            return opportunities
            
        except Exception as e:
            return {
                '3%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]},
                '5%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]},
                '8%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]},
                '10%': {'first_opportunity': None, 'daily_counts': [0, 0, 0]}
            }

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