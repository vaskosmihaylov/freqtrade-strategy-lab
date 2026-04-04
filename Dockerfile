FROM freqtradeorg/freqtrade:stable_freqaitorch

USER root

RUN mkdir -p /freqtrade/custom/freqaimodels \
    && chown -R ftuser:ftuser /freqtrade/custom

RUN pip install --no-cache-dir \
    catboost \
    hmmlearn \
    lightgbm \
    numba \
    plotly \
    sdnotify \
    wandb

COPY --chown=ftuser:ftuser LSTM/PyTorchLSTMModel.py /freqtrade/custom/freqaimodels/PyTorchLSTMModel.py
COPY --chown=ftuser:ftuser LSTM/PyTorchLSTMRegressor.py /freqtrade/custom/freqaimodels/PyTorchLSTMRegressor.py
COPY --chown=ftuser:ftuser LSTM/PyTorchLSTMModel.py /freqtrade/freqtrade/freqai/torch/PyTorchLSTMModel.py

USER ftuser
