# Strategy Collection

```
⚠️ Risk Disclaimer:
The strategies and content in this repository are provided for educational and reference purposes only. Using these strategies in live trading involves significant financial risk. Market conditions may change due to unforeseen factors, and past performance based on backtesting does not guarantee future results. Please ensure you fully understand the associated risks and seek professional advice if necessary. By using this repository, you acknowledge that you are solely responsible for any financial outcomes.
```

This repository contains a collection of strategies that I have created, tested, and optimized. It includes:

- Strategies that I have personally developed.
- Strategies that I have backtested.
- Optimized parameters for specific strategies.

Each folder within the repository corresponds to an individual strategy. It contains:

- The source code for the strategy.
- The configuration file for the strategy.
- The backtesting results for the strategy.

Feel free to explore and use these strategies as a reference or starting point for your own trading systems!

## FreqAI strategy status

Runnable with the current repository and Docker setup:

- `lstm` using `AlexStrategyFinalV9` + custom `PyTorchLSTMRegressor`
- `transformer` using `AlexStrategyFinalV9` + `PyTorchTransformerRegressor`
- `reinforcementlearning` using `freqai_rl_test_strat` + `ReinforcementLearner`
- `e0v1eai` using `E0V1EAI` + `LightGBMRegressorMultiTarget`
- `freqaihybrid` using `FreqAIHybridStrategy` + `LightGBMRegressorMultiTarget`
- `tm3multiclass` using `TM3MultiClass` + `CatboostFeatureSelectedMultiTargetClassifierV1`

Runnable as standard strategy services:

- `bitcoinml` using `BitcoinMLStrategy`

Runtime caveats:

- `bitcoinml` now points at the trimmed local support package under `BitcoinMLStrategy/local_ml` and its bundled model artifact.
- `tm3multiclass` now uses the repo-owned `TM3MultiClass/lib` and `TM3MultiClass/freqaimodels` runtime paths.

## Docker setup

The default `docker-compose.yml` is now CPU-safe for Linux servers without NVIDIA hardware.
For GPU hosts, use the override file `docker-compose.gpu.yml`.

Why this layout:

- one shared torch image for the PyTorch-backed FreqAI services
- one RL-specific image because Freqtrade ships RL support via a different base tag
- isolated runtime data per strategy under `docker-data/<strategy>`
- no hard GPU dependency in the default compose path

### Build image

```sh
docker compose build
```

Build the RL image too:

```sh
docker compose build reinforcementlearning
```

Optional GPU build/run path:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
```

### Run default backtests

```sh
docker compose run --rm lstm
docker compose run --rm transformer
docker compose run --rm reinforcementlearning
docker compose run --rm e0v1eai
docker compose run --rm freqaihybrid
docker compose run --rm bitcoinml
docker compose run --rm tm3multiclass
```

### Run in dry-run (paper trading)

Transformer:

```sh
docker compose run -d --name transformer-dryrun \
  -p 127.0.0.1:8080:8080 \
  transformer trade \
  --config /freqtrade/workspace/TRANSFORMER/config_freqai.json \
  --strategy-path /freqtrade/workspace/TRANSFORMER \
  --freqaimodel PyTorchTransformerRegressor
```

LSTM:

```sh
docker compose run --rm lstm trade \
  --config /freqtrade/workspace/LSTM/config_freqai.json \
  --strategy-path /freqtrade/workspace/LSTM \
  --freqaimodel PyTorchLSTMRegressor \
  --freqaimodel-path /freqtrade/custom/freqaimodels
```

Reinforcement learning:

```sh
docker compose run --rm reinforcementlearning trade \
  --config /freqtrade/workspace/ReinforcementLearning/config_freqai.json \
  --strategy-path /freqtrade/workspace/ReinforcementLearning \
  --freqaimodel ReinforcementLearner
```

E0V1EAI:

```sh
docker compose run --rm e0v1eai trade \
  --config /freqtrade/workspace/E0V1E/config_freqai.json \
  --strategy-path /freqtrade/workspace/E0V1E \
  --freqaimodel LightGBMRegressorMultiTarget
```

FreqAIHybrid:

```sh
docker compose run --rm freqaihybrid trade \
  --config /freqtrade/workspace/FreqAIHybridStrategy.py/config_freqai.json \
  --strategy-path /freqtrade/workspace/FreqAIHybridStrategy.py \
  --freqaimodel LightGBMRegressorMultiTarget
```

BitcoinML:

```sh
docker compose run --rm bitcoinml trade \
  --config /freqtrade/workspace/BitcoinMLStrategy/config.json \
  --strategy-path /freqtrade/workspace/BitcoinMLStrategy
```

TM3MultiClass:

```sh
docker compose run --rm tm3multiclass trade \
  --config /freqtrade/workspace/TM3MultiClass/config_freqai.json \
  --strategy-path /freqtrade/workspace/TM3MultiClass \
  --freqaimodel CatboostFeatureSelectedMultiTargetClassifierV1 \
  --freqaimodel-path /freqtrade/workspace/TM3MultiClass/freqaimodels
```

Useful lifecycle commands:

```sh
docker logs -f transformer-dryrun
docker rm -f transformer-dryrun
docker compose run --remove-orphans --rm transformer list-data --config /freqtrade/workspace/TRANSFORMER/config_freqai.json --show-timerange
```

### First run: download market data (required)

`lstm`, `transformer`, `reinforcementlearning`, `e0v1eai`, `freqaihybrid`, `bitcoinml`, and `tm3multiclass` use futures pairs.
Before backtests or dry-run, download candles into the mounted `user_data` volume.
As of **April 4, 2026**, start from `20250101` so you include all of 2025 plus current 2026 data.

Transformer:

```sh
START=20250101
END=$(date -d "+1 day" +%Y%m%d)
docker compose run --rm transformer download-data \
  --config /freqtrade/workspace/TRANSFORMER/config_freqai.json \
  --trading-mode futures \
  --timeframes 1h 2h 4h \
  --timerange ${START}-${END} \
  --prepend
```

LSTM:

```sh
START=20250101
END=$(date -d "+1 day" +%Y%m%d)
docker compose run --rm lstm download-data \
  --config /freqtrade/workspace/LSTM/config_freqai.json \
  --trading-mode futures \
  --timeframes 1h 2h 4h \
  --timerange ${START}-${END} \
  --prepend
```

Reinforcement learning:

```sh
START=20250101
END=$(date -d "+1 day" +%Y%m%d)
docker compose run --rm reinforcementlearning download-data \
  --config /freqtrade/workspace/ReinforcementLearning/config_freqai.json \
  --trading-mode futures \
  --timeframes 5m 15m 1h 4h \
  --timerange ${START}-${END} \
  --prepend
```

E0V1EAI / FreqAIHybrid:

```sh
START=20250101
END=$(date -d "+1 day" +%Y%m%d)
docker compose run --rm e0v1eai download-data \
  --config /freqtrade/workspace/E0V1E/config_freqai.json \
  --trading-mode futures \
  --timeframes 5m 15m 1h 4h \
  --timerange ${START}-${END} \
  --prepend
```

BitcoinML:

```sh
START=20250101
END=$(date -d "+1 day" +%Y%m%d)
docker compose run --rm bitcoinml download-data \
  --config /freqtrade/workspace/BitcoinMLStrategy/config.json \
  --trading-mode futures \
  --timeframes 1h \
  --timerange ${START}-${END} \
  --prepend
```

TM3MultiClass:

```sh
START=20250101
END=$(date -d "+1 day" +%Y%m%d)
docker compose run --rm tm3multiclass download-data \
  --config /freqtrade/workspace/TM3MultiClass/config_freqai.json \
  --trading-mode futures \
  --timeframes 15m 1h 4h \
  --timerange ${START}-${END} \
  --prepend
```

Notes:
- `lstm` and `transformer` use the static Bybit futures pairlist overlay from `configs/pairlist-static-bybit-futures-usdt.json`.
- `reinforcementlearning` is intentionally pinned to `BTC/USDT:USDT` to keep CPU load manageable on non-GPU hosts.
- `e0v1eai` and `freqaihybrid` are multi-target FreqAI strategies and use `LightGBMRegressorMultiTarget`.
- `bitcoinml` expects the bundled LightGBM/HMM artifact under `BitcoinMLStrategy/local_ml/models/bitcoin_model_v1.pkl`.
- `tm3multiclass` is configured from the recovered upstream source tree and uses Binance futures `ADA/USDT:USDT` by default.
- `--prepend` is important when files already exist and you want to backfill older candles.

Download a broader static Bybit futures universe (recommended once for initial FreqAI bootstrap):

```sh
START=20250101
END=$(date -d "+1 day" +%Y%m%d)
docker compose run --rm transformer download-data \
  --config /freqtrade/workspace/TRANSFORMER/config_freqai.json \
  --config /freqtrade/workspace/configs/pairlist-static-bybit-futures-usdt.json \
  --trading-mode futures \
  --timeframes 1h 2h 4h \
  --timerange ${START}-${END} \
  --prepend
```

Optional verification:

```sh
docker compose run --rm transformer list-data \
  --config /freqtrade/workspace/TRANSFORMER/config_freqai.json \
  --show-timerange
```

For GPU hosts, add the override file:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm lstm
```

### Troubleshooting `No data found. Terminating.`

If logs show warnings like:
- `No history for BTC/USDT:USDT, futures, 1h found`
- followed by `No data found. Terminating.`

then the container started correctly, but `./docker-data/<strategy>/data/bybit` does not contain the required futures candles yet (or only has spot data). Run the `download-data` command above, then run your `backtesting` or `trade` command again.

If you start `trade` and UI shows bot state `STOPPED`, this is a start-state issue (not a data issue). Start the bot from UI or set `"initial_state": "running"` in the strategy config.

If you see PyTorch `weights_only`/`UnpicklingError` warnings when FreqAI loads old checkpoints, this repo sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for `lstm`/`transformer` services in `docker-compose.yml`. If warnings persist, remove stale model artifacts under `docker-data/<strategy>/models` and let FreqAI retrain.

### Override with a custom command

Example (`transformer` backtest with explicit timerange):

```sh
docker compose run --rm transformer backtesting \
  --config /freqtrade/workspace/TRANSFORMER/config_freqai.json \
  --strategy AlexStrategyFinalV9 \
  --strategy-path /freqtrade/workspace/TRANSFORMER \
  --freqaimodel PyTorchTransformerRegressor \
  --timerange 20250101-20260331
```
