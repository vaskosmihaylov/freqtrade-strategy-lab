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

## Docker setup

This repository now includes a single `docker-compose.yml` for all three requested strategies:

- `lstm` (FreqAI + custom `PyTorchLSTMRegressor`)
- `transformer` (FreqAI + `PyTorchTransformerRegressor`)
- `e0v1e` (standard strategy)

Why one compose file:
- one shared image build
- less duplication and simpler maintenance
- isolated runtime data per strategy under `docker-data/<strategy>`

### Build image

```sh
docker compose build
```

### Run default backtests

```sh
docker compose run --rm lstm
docker compose run --rm transformer
docker compose run --rm e0v1e
```

### Run in dry-run (paper trading)

Transformer:

```sh
docker compose run --rm transformer trade \
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

E0V1E:

```sh
docker compose run --rm e0v1e trade \
  --config /freqtrade/workspace/E0V1E/config.json \
  --strategy E0V1E \
  --strategy-path /freqtrade/workspace/E0V1E
```

### First run: download market data (required)

`lstm` and `transformer` use Bybit **futures** pairs (for example `BTC/USDT:USDT`).
Before backtests or dry-run, download candles into the mounted `user_data` volume.
As of **February 26, 2026**, start from `20250101` so you include all of 2025 plus current 2026 data.

Transformer:

```sh
START=20250101
END=$(date +%Y%m%d)  # e.g. 20260226
docker compose run --rm transformer download-data \
  --config /freqtrade/workspace/TRANSFORMER/config_freqai.json \
  --trading-mode futures \
  --timeframes 1h 2h 4h \
  --timerange ${START}-${END}
```

LSTM:

```sh
START=20250101
END=$(date +%Y%m%d)  # e.g. 20260226
docker compose run --rm lstm download-data \
  --config /freqtrade/workspace/LSTM/config_freqai.json \
  --trading-mode futures \
  --timeframes 1h 2h 4h \
  --timerange ${START}-${END}
```

Optional verification:

```sh
docker compose run --rm transformer list-data \
  --config /freqtrade/workspace/TRANSFORMER/config_freqai.json \
  --show-timerange
```

### WSL2 + NVIDIA GPU notes

For your Windows + WSL2 Ubuntu setup (RTX 4080 Super), ensure:

1. Latest NVIDIA Windows driver is installed (WSL CUDA support enabled).
2. Docker Desktop has WSL integration enabled for your Ubuntu distro.
3. NVIDIA Container Toolkit is available to Docker in WSL.
4. Keep this repo on the WSL filesystem (for example `~/freqtrade-strategy-lab`), not under `/mnt/c/...`, to avoid bind-mount and I/O issues.

Quick check:

```sh
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

`lstm` and `transformer` services are configured to request GPU devices.

### Troubleshooting `No data found. Terminating.`

If logs show warnings like:
- `No history for BTC/USDT:USDT, futures, 1h found`
- followed by `No data found. Terminating.`

then the container started correctly, but `./docker-data/<strategy>/data/bybit` does not contain the required futures candles yet (or only has spot data). Run the `download-data` command above, then run your `backtesting` or `trade` command again.

### Override with a custom command

Example (`E0V1E` hyperopt):

```sh
docker compose run --rm e0v1e hyperopt --hyperopt-loss SharpeHyperOptLossDaily --spaces buy --strategy E0V1E --config /freqtrade/workspace/E0V1E/config.json -e 1500 -j 2 --analyze-per-epoch --strategy-path /freqtrade/workspace/E0V1E
```
