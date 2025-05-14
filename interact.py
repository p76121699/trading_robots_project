import numpy as np
import pandas as pd
import torch
import os
from collections import deque
from utils.Stock_API import *
from datetime import datetime, timedelta
import pandas_ta as ta
from algs.drl_agent import Actor
from argparse import Namespace
from config.env.env_config import env_config

# 1. 初始化 agent 模型
class TradingExecutor:
    def __init__(self, model_path, stock_ids, user_info, window_size=30, action_gap=10, pos_save_path="position_history.npy"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stock_ids = stock_ids
        self.N = len(stock_ids)
        self.window_size = window_size
        self.num_actions = 11
        self.action_gap = action_gap
        self.pos_save_path = pos_save_path

        state_dim = self.N * (1 + 6)  # position + 6 個技術指標
        self.actor = Actor(state_dim, self.N, self.num_actions)
        self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
        self.actor.to(self.device)
        self.actor.eval()

        # 初始化用戶
        self.user = Stock_API(user_info[0], user_info[1])

        # 初始化 position history
        self.pos_history = self.load_or_initialize_position()

    def load_or_initialize_position(self):
        if os.path.exists(self.pos_save_path):
            return deque(np.load(self.pos_save_path).tolist(), maxlen=self.window_size)
        else:
            return deque([np.zeros(self.N, dtype=np.float32)] * (self.window_size - 1), maxlen=self.window_size)

    def save_position_history(self):
        np.save(self.pos_save_path, np.stack(self.pos_history, axis=0))

    def fetch_position_array(self):
        user_stocks = self.user.Get_User_Stocks()
        pos_dict = {s['stock_code_id']: s['shares'] for s in user_stocks}
        
        position_array = np.array([pos_dict.get(sid, 0) for sid in self.stock_ids], dtype=np.float32)
        
        return position_array

    def fetch_price_features(self, sid, start_date, end_date):
        raw = Stock_API.Get_Stock_Informations(sid, start_date, end_date)
        df = pd.DataFrame(raw)
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df = df.set_index('date').sort_index()

        df.ta.macd(close='close', append=True)
        df.ta.rsi(close='close', append=True)
        df.ta.cci(append=True)
        df.ta.adx(append=True)
        df.ta.atr(append=True)
        
        return df[['close', 'MACD_12_26_9', 'RSI_14', 'CCI_14_0.015', 'ADX_14', 'ATRr_14']].dropna()

    def get_today_state(self, today_str):
        today = datetime.strptime(today_str, "%Y%m%d")
        start_date = (today - timedelta(days=50)).strftime("%Y%m%d")
        end_date = today_str

        price_features = []
        for sid in self.stock_ids:
            feats = self.fetch_price_features(sid, start_date, end_date)
            feats = feats.iloc[-self.window_size:].values  # [W, 6]
            price_features.append(feats)
        price_features = np.stack(price_features, axis=1)  # [W, N, 6]

        if len(self.pos_history) < self.window_size:
            raise ValueError("position queue 不足，請先填滿。")
        pos_array = np.stack(list(self.pos_history), axis=0)  # [W, N]

        state = np.concatenate([pos_array[:, :, None], price_features], axis=-1)  # [W, N, 7]
        return state.reshape(self.window_size, -1)  # [W, N*7]

    def decode_action(self, action_idx):
        actions = []
        for idx in action_idx:
            if idx < self.num_actions // 2:
                action_type = 2  # Sell
                quantity = (self.num_actions // 2 - idx) * self.action_gap
            elif idx > self.num_actions // 2:
                action_type = 1  # Buy
                quantity = (idx - self.num_actions // 2) * self.action_gap
            else:
                action_type = 0  # Hold
                quantity = 0
            actions.append((action_type, quantity))
        return actions

    def trade(self, today_str):
        # 1. 取得今日 position，並 append 到 queue
        pos_array = self.fetch_position_array()
        
        self.pos_history.append(pos_array)
        self.save_position_history()

        if len(self.pos_history) < self.window_size:
            print("等待 position queue 填滿中...")
            return

        # 2. 取得狀態
        state = self.get_today_state(today_str)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 7*N*features]

        # 3. Actor 推論
        with torch.no_grad():
            logits = self.actor(state)  # [1, N, A]
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample().squeeze(0).cpu().numpy()  # [N]
        actions = self.decode_action(action_idx)

        # 4. 取得昨日收盤價
        yesterday = (datetime.strptime(today_str, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        close_price = {}
        for sid in self.stock_ids:
            info = Stock_API.Get_Stock_Informations(sid, yesterday, yesterday)
            close_price[sid] = info[0]['close'] if info else 0

        # 5. 發出下單指令
        for i, sid in enumerate(self.stock_ids):
            act_type, qty = actions[i]
            if qty == 0 or close_price[sid] == 0:
                continue
            if act_type == 1:
                print(f"買入 {sid} × {qty} 張 @ {close_price[sid]}")
                self.user.Buy_Stock(sid, qty, close_price[sid])
            elif act_type == 2:
                print(f"賣出 {sid} × {qty} 張 @ {close_price[sid]}")
                self.user.Sell_Stock(sid, qty, close_price[sid])


if __name__ == "__main__":
    args = Namespace(**env_config)

    user_info = ['P76123456', '3456'] # 個人帳戶
    
    trader = TradingExecutor(
        model_path="./models/reinforce_model",  # 請確認模型存在此處
        stock_ids=args.stock_codes,
        user_info=user_info,
        window_size=7,
        action_gap=1,
        pos_save_path="./position_history_for_interact.npy" # 第一次運行時會自動建立
    )

    today = "20250514"
    trader.trade(today)
