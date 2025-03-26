import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from src.data.stock_manager import StockManager
from datetime import datetime
import math

class StockGUI:
    def __init__(self):
        self.root = ttk.Window(themename="cyborg")
        self.root.title("股票管理系统")
        self.root.geometry("1200x800")
        
        # 创建主容器
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # 创建顶部工具栏
        self.create_toolbar()
        
        # 创建滚动画布
        self.create_scrollable_frame()
        
        # 初始化股票管理器
        self.stock_manager = StockManager()
        
        # 加载股票数据
        self.load_stocks()

    def create_toolbar(self):
        toolbar = ttk.Frame(self.main_container)
        toolbar.pack(fill=X, pady=(0, 10))
        
        # 添加股票按钮
        add_btn = ttk.Button(
            toolbar,
            text="添加股票",
            style="info.TButton",
            command=self.show_add_dialog
        )
        add_btn.pack(side=RIGHT)
        
        # 查看历史记录按钮
        history_btn = ttk.Button(
            toolbar,
            text="历史记录",
            style="secondary.TButton",
            command=self.show_history
        )
        history_btn.pack(side=RIGHT, padx=5)

        # 添加刷新按钮
        refresh_btn = ttk.Button(
            toolbar,
            text="刷新数据",
            style="primary.TButton",
            command=self.refresh_data
        )
        refresh_btn.pack(side=RIGHT, padx=5)

    def create_scrollable_frame(self):
        # 创建画布和滚动条
        self.canvas = tk.Canvas(self.main_container)
        scrollbar = ttk.Scrollbar(self.main_container, orient=VERTICAL, command=self.canvas.yview)
        
        # 创建可滚动框架
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # 在画布上创建窗口
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor=NW)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

    def create_stock_card(self, stock_data):
        # 创建股票卡片框架
        card = ttk.Frame(self.scrollable_frame, style="dark.TFrame")
        card.pack(fill=X, padx=5, pady=5)
        
        # 第一行：股票代码和名称
        header_frame = ttk.Frame(card)
        header_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(
            header_frame,
            text=f"{stock_data['code']} {stock_data['name']}",
            style="info.TLabel",
            font=("Arial", 12, "bold")
        ).pack(side=LEFT)
        
        # 删除按钮
        ttk.Button(
            header_frame,
            text="×",
            style="danger.TButton",
            command=lambda: self.delete_stock(stock_data['code'])
        ).pack(side=RIGHT)
        
        # 第二行：基本信息
        info_frame = ttk.Frame(card)
        info_frame.pack(fill=X, padx=5, pady=2)
        info_text = f"现价：{stock_data['current_price']:.2f} 成本：{stock_data['cost']:.2f} " \
                   f"数量：{stock_data['quantity']} 总成本：{stock_data['total_cost']:.2f} " \
                   f"入手日期：{stock_data['register_date'].strftime('%Y-%m-%d')}"
        ttk.Label(info_frame, text=info_text).pack(fill=X)
        
        # 目标价和止损价信息
        targets = [3, 5, 8, 10]
        for target in targets:
            target_frame = ttk.Frame(card)
            target_frame.pack(fill=X, padx=5, pady=1)
            target_price = stock_data['cost'] * (1 + target/100)
            has_opportunity = "有机会" if stock_data['current_price'] >= target_price else "暂无机会"
            opportunity_style = "success.TLabel" if has_opportunity == "有机会" else "secondary.TLabel"
            
            ttk.Label(
                target_frame,
                text=f"{target}%目标价：{target_price:.2f}",
            ).pack(side=LEFT)
            ttk.Label(
                target_frame,
                text=has_opportunity,
                style=opportunity_style
            ).pack(side=RIGHT)
        
        # 止损价信息
        stop_loss_frame = ttk.Frame(card)
        stop_loss_frame.pack(fill=X, padx=5, pady=1)
        stop_loss_price = stock_data['cost'] * 0.95
        is_stop_loss = "已触发" if stock_data['current_price'] <= stop_loss_price else "未触发"
        stop_loss_style = "danger.TLabel" if is_stop_loss == "已触发" else "success.TLabel"
        
        ttk.Label(
            stop_loss_frame,
            text=f"止损价：{stop_loss_price:.2f}"
        ).pack(side=LEFT)
        ttk.Label(
            stop_loss_frame,
            text=is_stop_loss,
            style=stop_loss_style
        ).pack(side=RIGHT)

    def show_add_dialog(self):
        dialog = ttk.Toplevel(self.root)
        dialog.title("添加股票")
        dialog.geometry("300x200")
        
        # 创建输入框
        ttk.Label(dialog, text="股票代码：").pack(pady=5)
        code_entry = ttk.Entry(dialog)
        code_entry.pack(pady=5)
        
        ttk.Label(dialog, text="购买价格：").pack(pady=5)
        price_entry = ttk.Entry(dialog)
        price_entry.pack(pady=5)
        
        ttk.Label(dialog, text="购买数量：").pack(pady=5)
        quantity_entry = ttk.Entry(dialog)
        quantity_entry.pack(pady=5)
        
        # 添加按钮
        def add_stock():
            try:
                code = code_entry.get()
                price = float(price_entry.get())
                quantity = int(quantity_entry.get())
                
                success, message = self.stock_manager.add_stock(code, price, quantity)
                if success:
                    messagebox.showinfo("成功", "股票添加成功！")
                    dialog.destroy()
                    self.load_stocks()  # 重新加载股票列表
                else:
                    messagebox.showerror("错误", f"添加失败：{message}")
            except ValueError:
                messagebox.showerror("错误", "请输入有效的价格和数量！")
        
        ttk.Button(
            dialog,
            text="添加",
            style="success.TButton",
            command=add_stock
        ).pack(pady=10)

    def delete_stock(self, code):
        if messagebox.askyesno("确认", f"确定要删除股票 {code} 吗？"):
            success, message = self.stock_manager.delete_stock(code)
            if success:
                messagebox.showinfo("成功", "股票删除成功！")
                self.load_stocks()  # 重新加载股票列表
            else:
                messagebox.showerror("错误", f"删除失败：{message}")

    def show_history(self):
        history_window = ttk.Toplevel(self.root)
        history_window.title("删除历史记录")
        history_window.geometry("800x600")
        
        # 创建表格
        columns = ("股票代码", "股票名称", "删除时价格", "成本", "数量", "总成本", "删除时间")
        tree = ttk.Treeview(history_window, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # 获取历史记录
        history_records = self.stock_manager.get_delete_history()
        
        # 添加数据到表格
        for record in history_records:
            tree.insert("", END, values=(
                record['code'],
                record['name'],
                f"{record['price']:.2f}",
                f"{record['cost']:.2f}",
                record['quantity'],
                f"{record['total_cost']:.2f}",
                record['delete_time'].strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(history_window, orient=VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

    def load_stocks(self):
        # 清除现有的股票卡片
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # 获取并显示所有股票
        stocks = self.stock_manager.list_all_stocks()
        for stock in stocks:
            self.create_stock_card(stock)

    def refresh_data(self):
        """刷新所有股票数据"""
        try:
            self.load_stocks()  # 重新加载所有股票数据
            messagebox.showinfo("成功", "数据刷新成功！")
        except Exception as e:
            messagebox.showerror("错误", f"刷新失败：{str(e)}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    # 初始化数据库
    StockManager.init_database()
    # 启动GUI
    app = StockGUI()
    app.run() 