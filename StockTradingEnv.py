from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict
from utils.Stock_API import *
from collections import deque
import pandas_ta as ta

class StockTradingEnvMulti:
    HOLD, BUY, SELL = 0, 1, 2
    
    def __init__(self, args):
        self.stock_codes = getattr(args, "stock_codes", [
            "9105", "2365", "2374", "2313", "2498",
            "2603", "2609", "2615", "2610", "2618",
            "3481", "2409", "2344", "3035", "2408",
            "1605", "2002", "1101", "1102", "2412"
        ])
        self.N = len(self.stock_codes)

        self.start_date = datetime.strptime(getattr(args, "start_date", "20180101"), "%Y%m%d")
        self.end_date = datetime.strptime(getattr(args, "end_date", "20241231"), "%Y%m%d")

        self.window_size = getattr(args, "window_size", 30)
        self.initial_cash = float(getattr(args, "initial_cash", 100_000_000))

        self.transaction_cost = getattr(args, "transaction_cost", 1e-4)
        self.decay_rate = getattr(args, "decay_rate", 1e-4)
        self.reward_scaling = getattr(args, "reward_scaling", 1e-2)

        # 資料載入
        self.data_panel, self.dates = self._load_data_multi()
        self.close_mat = self.data_panel[:, :, 0]  # shape: (days, N)

        # 動態變數初始化
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment state and return the first observation."""
        self.day_ptr: int = self.window_size  # 窗口右端第一筆可交易日
        self.cash_balance: float = self.initial_cash
        self.positions: np.ndarray = np.zeros(self.N, dtype=np.float32)
        self.avg_buy_prices: np.ndarray = np.zeros(self.N, dtype=np.float32)
        self.holding_days: np.ndarray = np.zeros(self.N, dtype=np.int32)
        self.done: bool = False
        
        self.pos_queue: deque = deque([self.positions.copy() for _ in range(self.window_size)], maxlen=self.window_size)
        
        return self._get_state()
    
    def _load_data_multi(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (panel, aligned_dates).

        panel shape = (days, N, features) where features ==
        [close, MACD, RSI, CCI, ADX, ATR]
        """
        frames: List[pd.DataFrame] = []
        for code in self.stock_codes:
            df = self._load_single(code)
            frames.append(df)

        # 透過 inner join 只保留所有股票都有資料的日期
        merged: pd.DataFrame = pd.concat(frames, axis=1, keys=self.stock_codes, join="inner")
        merged.sort_index(inplace=True)

        # 轉成 3‑D numpy array
        days = merged.shape[0]
        panel = merged.values.reshape(days, self.N, -1).astype(np.float32)
        dates = merged.index.to_numpy()
        return panel, dates
    
    def _load_single(self, stock_code: str) -> pd.DataFrame:
        extra_days = 50
        ext_start = (self.start_date - pd.Timedelta(days=extra_days)).strftime("%Y%m%d")
        raw = Stock_API.Get_Stock_Informations(stock_code, ext_start, self.end_date.strftime("%Y%m%d"))
        
        df = pd.DataFrame(raw)
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df.set_index('date', inplace=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        df.ta.macd(close='close', append=True)
        df.ta.rsi(close='close', append=True)
        df.ta.cci(append=True)
        df.ta.adx(append=True)
        df.ta.atr(append=True)

        # 用你原本要的欄位名整理一下對應欄位
        df = df.rename(columns={
            'MACD_12_26_9': 'MACD',
            'RSI_14': 'RSI',
            'CCI_14_0.015': 'CCI',
            'ADX_14': 'ADX',
            'ATRr_14': 'ATR',
        })

        # 篩選正式模擬期間
        df = df.loc[self.start_date : self.end_date]
        return df[['close', 'MACD', 'RSI', 'CCI', 'ADX', 'ATR']].dropna()

    def _get_state(self) -> np.ndarray:
        """Get current observation state for the agent.

        Returns
        -------
        state : ndarray, shape (window_size, N * 7)
            Each row represents a time step in the past `window_size` days.
            For each of the N stocks, the 7 features per day include:
                - current position (1 feature)
                - technical indicators or price features (6 features)
            So each day's feature is flattened to a 1D array of length N * 7.
            If the environment is done, returns a zero array of the same shape.
        """
        
        if self.done:
            feature_dim = (self.N * (1 + 6))
            return np.zeros((self.window_size, feature_dim), dtype=np.float32)

        pos_array = np.stack(self.pos_queue, axis=0)
        price_array = self.data_panel[self.day_ptr - self.window_size : self.day_ptr]

        pos_feat_array = np.concatenate([
            pos_array[:, :, np.newaxis],
            price_array,
        ], axis=-1)
        pos_feat_flat = pos_feat_array.reshape(self.window_size, -1)
        
        return pos_feat_flat.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one trading day.

        Parameters
        ----------
        action : ndarray, shape (N, 2)
            action[:,0] = {0,1,2}, action[:,1] = qty (int)
        """

        action = np.asarray(action)
        assert action.shape == (self.N, 2), "Action must be (N,2)."
        
        realised_pnl = 0.0
        fee_total = 0.0
        
        exec_prices = self.close_mat[self.day_ptr]  # (N,)
        quantities = action[:, 1]
        prev_total_asset = self.cash_balance + (self.positions * self.close_mat[self.day_ptr - 1]).sum()
        # BUY -----------------------------------------------------------------
        buy_mask = action[:, 0] == self.BUY
        if buy_mask.any():
            cost = exec_prices[buy_mask] * quantities[buy_mask]
            fee = cost * self.transaction_cost
            total_cost = cost + fee
            affordable = total_cost <= self.cash_balance
            if affordable.sum() > 0:
                idx = np.where(buy_mask)[0][affordable]
                self.cash_balance -= total_cost[affordable].sum()
                fee_total += fee[affordable].sum()

                # 更新平均成本
                new_qty = quantities[idx]
                new_price = exec_prices[idx]
                pos_before = self.positions[idx]
                cost_before = self.avg_buy_prices[idx] * pos_before
                pos_after = pos_before + new_qty
                self.avg_buy_prices[idx] = (cost_before + new_price * new_qty) / pos_after
                self.positions[idx] = pos_after

        # SELL ----------------------------------------------------------------
        sell_mask = action[:, 0] == self.SELL
        if sell_mask.any():
            sell_qty = np.minimum(quantities[sell_mask], self.positions[sell_mask])
            
            fail_sell = (sell_qty == 0)
            if fail_sell.any():
                sell_fail_penalty = -0.1 * fail_sell.sum()
            else:
                sell_fail_penalty = 0
            revenue = exec_prices[sell_mask] * sell_qty
            fee = revenue * self.transaction_cost
            fee_total += fee.sum()
            pnl = (exec_prices[sell_mask] - self.avg_buy_prices[sell_mask]) * sell_qty
            realised_pnl += pnl.sum()

            self.cash_balance += (revenue - fee).sum()
            self.positions[sell_mask] -= sell_qty
            self.avg_buy_prices[sell_mask][self.positions[sell_mask] == 0] = 0.0
            self.holding_days[sell_mask] = 0

        # HOLD update ---------------------------------------------------------
        self.holding_days[self.positions > 0] += 1
        time_cost = (self.decay_rate * self.holding_days * self.positions * exec_prices).sum()
        current_total_asset = self.cash_balance + (self.positions * self.close_mat[self.day_ptr]).sum()
        reward = -0.001
        # ---------- reward ---------- #
        reward += (current_total_asset - prev_total_asset) * self.reward_scaling  #兩個連續時間的總資本差作為獎勵
        reward += sell_fail_penalty
        # ---------- episode update ---------- #
        self.day_ptr += 1
        if self.day_ptr >= len(self.dates):
            self.done = True
            
        self.pos_queue.append(self.positions.copy())
        
        next_state = self._get_state()
        
        info = {
            "cash": self.cash_balance,
            "positions": self.positions.copy(),
            "realised_pnl": realised_pnl,
            "fee": fee_total,
            "time_cost": time_cost,
        }
        return next_state, float(reward), self.done, info

    
    def render(self):
        """
        畫出當前的價格走勢與當前日期標記（可選擇性實作）。
        """
        pass

if __name__ == "__main__":
    env = StockTradingEnvMulti(
        stock_codes=["9105", "2365", "2374", "2313", "2498",
                     "2603", "2609", "2615", "2610", "2618",
                     "3481", "2409", "2344", "3035", "2408",
                     "1605", "2002", "1101", "1102", "2412"],  # 20 檔示例
        start_date="20180101",
        end_date="20250422",
    )

    state = env.reset()
    print("state shape:", state.shape)

    done = False
    while not done:
        dummy_action = np.zeros((env.N, 2), dtype=int)  # 全 HOLD
        state, reward, done, info = env.step(dummy_action)
        break
    env.liquidate()
    print("Final cash:", info['cash'])