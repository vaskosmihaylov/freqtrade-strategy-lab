
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import os
from typing import List, Tuple, Dict, Any

class BitcoinTradingModel:
    def __init__(self):
        self.hmm_model = None
        self.lgbm_model = None
        self.scaler = StandardScaler()
        self.feature_cols = [] # 已使用的特徵列
        self.regime_map = {}  # Map states to volatility levels

    # --- 特徵工程 (複製/改編邏輯) ---
    def calculate_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        計算技術指標並將其添加到 DataFrame 中。
        :param df: 原始 K 線數據
        :param drop_na: 是否刪除含有空值的行
        :return: 帶有特徵的 DataFrame
        """
        df = df.copy()
        
        # 確保數據類型正確
        for col in ['open', 'high', 'low', 'close', 'vol']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # 基礎移動平均線 (MA)
        for window in [7, 25, 99]:
            df[f'ma{window}'] = df['close'].rolling(window=window).mean()

        # 相對強弱指標 (RSI)
        self._append_rsi(df, 14)
        
        # 平均真實波幅 (ATR)
        self._append_atr(df, 14)

        # 平滑異同移動平均線 (MACD)
        self._append_macd(df)

        # 布林帶 (Bollinger Bands)
        self._append_bbands(df)
        
        # 目標：下一階段收益是否超過閾值 (0.5%)
        # 1 表示上漲超過 0.5%，0 表示其他狀況 (平盤或下跌)
        threshold = 0.005
        df['target'] = ((df['close'].shift(-1) > df['close'] * (1 + threshold))).astype(int)

        if drop_na:
            return df.dropna()
        return df

    def _append_rsi(self, df: pd.DataFrame, window: int):
        """計算相對強弱指標 (RSI)"""
        # 計算價格變動
        delta = df['close'].diff()
        # 分別獲取上漲和下跌的幅度
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        # 使用指數加權移動平均 (EWMA) 計算平均上漲和平均下跌
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        # 計算相對強度 (RS)
        rs = avg_gain / avg_loss
        # 計算 RSI 並存入 DataFrame
        df['rsi'] = 100 - (100 / (1 + rs))

    def _append_atr(self, df: pd.DataFrame, window: int):
        """計算平均真實波幅 (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        # 計算三種真實波幅 (TR)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        # 取三者中的最大值作為當前的 TR
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # 使用 EWMA 計算平滑後的 ATR
        df['atr'] = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    def _append_macd(self, df: pd.DataFrame, fast=12, slow=26, signal=9):
        """計算平滑異同移動平均線 (MACD)"""
        # 計算快線和慢線的指數移動平均 (EMA)
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        # MACD 線 = 快線 - 慢線
        macd = exp1 - exp2
        # 信號線 (Signal Line) = MACD 線的 EMA
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        # 存入 DataFrame
        df['macd'] = macd
        df['macd_signal'] = signal_line
        df['macd_hist'] = macd - signal_line

    def _append_bbands(self, df: pd.DataFrame, window=20, num_std=2):
        """計算布林帶 (Bollinger Bands)"""
        # 計算滾動平均值 (中軌) 和滾動標準差
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        # 上軌 = 中軌 + (標準差 * 倍數)
        df['bb_upper'] = rolling_mean + (rolling_std * num_std)
        # 下軌 = 中軌 - (標準差 * 倍數)
        df['bb_lower'] = rolling_mean - (rolling_std * num_std)
    
    # --- 1. 使用 HMM 進行市場狀態 (Regime) 檢測 ---
    def train_hmm(self, df: pd.DataFrame, n_components=3):
        """
        基於收益率和波動率訓練 HMM，以識別市場狀態。
        :param df: 帶有特徵的 DataFrame
        :param n_components: 隱藏狀態的數量 (預設為 3)
        """
        # 準備 HMM 數據 (收益率和價格區間)
        if len(df) < 100:
             print("數據不足，無法訓練 HMM")
             return

        # 計算收益率 (pct_change)
        returns = df['close'].pct_change().dropna()
        # 對齊索引
        df_aligned = df.loc[returns.index]
        
        # 計算價格區間指標 (最高價減最低價除以收盤價)
        range_stat = (df_aligned['high'] - df_aligned['low']) / df_aligned['close']
        
        # 合併特徵矩陣 X
        X = np.column_stack([returns.values, range_stat.values])
        
        # 訓練 HMM 模型
        # 使用 GaussianHMM，隱藏狀態數為 n_components
        self.hmm_model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
        self.hmm_model.fit(X)
        
        # 分析各個狀態，以確定哪些是高波動/不適合交易的
        # 我們假設收益率的變異數愈大 = 波動率愈高
        # self.hmm_model.covars_ 形狀為 (n_components, n_features, n_features)
        variances = [np.mean(np.diag(self.hmm_model.covars_[i])) for i in range(n_components)]
        
        # 按照波動率對狀態進行排名 (0 = 最低波動, n-1 = 最高波動)
        sorted_indices = np.argsort(variances)
        
        # 將原始狀態索引映射到排名
        self.regime_map = {original_idx: rank for rank, original_idx in enumerate(sorted_indices)}
        
        print(f"各狀態的 HMM 變異數: {variances}")
        print(f"狀態映射表 (狀態 -> 排名 (0=低波動)): {self.regime_map}")

    def get_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """
        預測輸入數據的市場狀態。
        :param df: 帶有 K 線數據的 DataFrame
        :return: 包含每個數據點狀態索引的數組
        """
        if self.hmm_model is None:
            raise ValueError("HMM 模型尚未訓練")
            
        returns = df['close'].pct_change()
        # 對齊索引
        df_aligned = df.loc[returns.index]
        
        range_stat = (df_aligned['high'] - df_aligned['low']) / df_aligned['close']
        
        # HMM 需要乾淨的數據，因此我們刪掉 pct_change 產生的 NaN (通常是索引 0)
        if len(returns) == 0:
            return np.array([])

        # 處理無效數據
        valid_mask = ~np.isnan(returns)
        returns_valid = returns[valid_mask]
        range_valid = range_stat[valid_mask]
        
        if len(returns_valid) == 0:
             return np.zeros(len(df)) # 回退方案
             
        X = np.column_stack([returns_valid.values, range_valid.values])
        
        # 進行狀態預測
        hidden_states = self.hmm_model.predict(X)
        
        # 將預測結果映射回原始 DataFrame 的長度
        # 創建一個初始值為 -1 (未知) 的 Series
        s_states = pd.Series(hidden_states, index=returns[valid_mask].index)
        
        # 填充到與原始 df 相同的索引中
        df_states = pd.Series(index=df.index, dtype=float)
        df_states.update(s_states)
        
        return df_states.fillna(-1).values.astype(int)

    def filter_regime(self, df: pd.DataFrame, max_vol_rank=1) -> pd.DataFrame:
        """
        過濾掉屬於高波動狀態的行。
        :param df: 帶有 regime 數據的 DataFrame
        :param max_vol_rank: 允許的最大波動率排名 (0=最低，排名愈高代表波動愈劇烈)
                             例如 n_components=3 時，排名為 0, 1, 2。max_vol_rank=1 會保留 0 和 1。
        :return: 過濾後的 DataFrame
        """
        # 獲取每個數據點的市場狀態
        states = self.get_regimes(df)
        df_filtered = df.copy()
        df_filtered['regime'] = states
        
        # 將狀態映射到排名
        # 如果狀態為 -1 (未知)，則映射為 999
        df_filtered['regime_rank'] = df_filtered['regime'].apply(lambda x: self.regime_map.get(x, 999))
        
        # 過濾：僅保留排名小於或等於 max_vol_rank 的狀態
        # 同時排除排名為 999 (未知/第一行) 的數據
        return df_filtered[df_filtered['regime_rank'] <= max_vol_rank]

    # --- 2. (已移除) 隨機森林特徵篩選 ---
    # 根據建議，移除隨機森林篩選，直接讓 LightGBM 處理特徵。

    # --- 3. 使用 LightGBM 進行預測訓練 ---
    def train_lgbm(self, df: pd.DataFrame, target_col='target'):
        """
        基於所有特徵訓練 LightGBM 模型。
        :param df: 已經過濾市場狀態的 DataFrame
        :param target_col: 目標列名稱
        """
        # 定義特徵列 (排除非特徵列)
        exclude_cols = ['start_time', 'end_time', 'open', 'high', 'low', 'close', 'vol', 'number_of_trade', 'target', 'regime', 'regime_rank']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        print(f"Training on features: {self.feature_cols}")

        if not self.feature_cols:
             raise ValueError("No features found for training.")

        # 提取特徵和目標
        X = df[self.feature_cols]
        y = df[target_col]
        
        # 劃分訓練集和測試集 (不打亂順序，因為是時間序列數據)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # 特徵標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 創建 LightGBM 數據集
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        # 設定 LightGBM 參數 (User Optimized)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,          # 降低葉子數量，防止過擬合
            'max_depth': 4,            # 限制樹深度
            'learning_rate': 0.01,     # 降低學習率
            'feature_fraction': 0.7,   # 特徵採樣
            'bagging_fraction': 0.7,   # 數據採樣
            'bagging_freq': 5,
            'lambda_l1': 0.1,          # L1 正則化
            'lambda_l2': 0.1,          # L2 正則化
            'min_data_in_leaf': 20,
            'verbose': -1
        }
        
        # 開始訓練
        self.lgbm_model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(50)
            ]
        )


    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts probabilities using the trained LightGBM model.
        """
        if not self.lgbm_model:
            raise ValueError("LGBM model not trained")
            
        if not self.feature_cols:
             raise ValueError("Feature columns not defined")

        # Extract features
        # Check if features exist
        missing_cols = set(self.feature_cols) - set(df.columns)
        if missing_cols:
             print(f"Warning: Missing columns {missing_cols}, filling with 0")
             for col in missing_cols:
                 df[col] = 0
        
        X = df[self.feature_cols].copy()
        
        # Handle missings
        X = X.fillna(0) 
        
        X_scaled = self.scaler.transform(X)
        return self.lgbm_model.predict(X_scaled)

    def full_pipeline_train(self, df: pd.DataFrame):
        """
        執行完整訓練流程的便捷方法：
        1. 特徵計算 (Calculate Features)
        2. 訓練 HMM (Train HMM)
        3. 過濾市場狀態 (Filter Regime)
        4. 篩選特徵 (Select Features)
        5. 訓練 LightGBM (Train LightGBM)
        :param df: 原始 K 線數據
        """
        print("正在計算特徵...")
        df = self.calculate_features(df)
        
        print("正在訓練 HMM...")
        self.train_hmm(df)
        
        print("正在過濾市場狀態...")
        df_filtered = self.filter_regime(df, max_vol_rank=1)
        print(f"狀態過濾後，數據從 {len(df)} 行減少到 {len(df_filtered)} 行。")
        
        print("正在訓練 LightGBM (直接使用所有特徵)...")
        self.train_lgbm(df_filtered)
        print("訓練完成。")

    def save_model(self, path: str):
        joblib.dump({
            'hmm_model': self.hmm_model,
            'lgbm_model': self.lgbm_model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'regime_map': self.regime_map
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
            
        data = joblib.load(path)
        self.hmm_model = data['hmm_model']
        self.lgbm_model = data['lgbm_model']
        self.scaler = data['scaler']
        self.feature_cols = data['feature_cols']
        self.regime_map = data['regime_map']
        print(f"Model loaded from {path}")
