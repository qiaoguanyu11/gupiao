# 股票管理系统

这是一个基于Python的股票管理系统，用于跟踪和分析股票交易机会。

## 功能特点

- 股票添加和管理
- 实时股价监控
- T+1交易模式下的出手机会分析
- 多档位收益率监控（3%、5%、8%、10%）
- 止损点分析
- 实时数据获取和分析

## 技术栈

- Python 3.x
- MySQL
- akshare（股票数据API）
- SQLAlchemy（数据库ORM）

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/qiaoguanyu11/gupiao.git
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置数据库：
- 确保MySQL服务已启动
- 创建数据库：stock_tracker
- 修改数据库连接配置（如需要）

## 使用方法

运行主程序：
```bash
python main.py
```

主要功能：
1. 添加股票
2. 查看股票列表
3. 删除股票
4. 测试数据源
5. 退出

## 注意事项

- 系统使用T+1交易规则
- 交易费用已考虑（买入0.04%，卖出0.4%）
- 实时数据每分钟更新
- 止损点设置在实际成本的95% 