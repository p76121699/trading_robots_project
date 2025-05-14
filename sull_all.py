from utils.Stock_API import *
from datetime import datetime, timedelta

# 初始化使用者
user = Stock_API('P76123456', '3456')

# 取得目前持股
user_stocks = user.Get_User_Stocks()

if not user_stocks:
    print("目前無持股，無需清倉。")
    exit()

# 取得今日與昨日日期
today = datetime.today()
yesterday_str = (today - timedelta(days=1)).strftime("%Y%m%d")

print("========== 清倉開始 ==========")

for stock in user_stocks:
    stock_id = stock['stock_code_id']
    quantity = stock['shares']

    # 取得昨日收盤價
    info = Stock_API.Get_Stock_Informations(stock_id, yesterday_str, yesterday_str)
    if not info:
        print(f"無法取得 {stock_id} 的昨日收盤價，跳過。")
        continue
    price = info[0]['close']

    print(f"賣出 {stock_id} × {quantity} 張 @ {price}")
    result = user.Sell_Stock(stock_id, quantity, price)


print("========== 清倉結束 ==========")
