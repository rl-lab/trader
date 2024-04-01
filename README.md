# trader


##  Install the first time

`ta-lib`  for calculating technical indicators
```bash

apt-get update

# python3 and pip are optional is system already have them
apt-get install -y build-essential wget gcc g++ make cmake build-essential python3 python3-pip

# Download and extract TA-Lib source
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz

# Compile and install TA-Lib
cd ta-lib
./configure --prefix=/usr
make
make install

# Clean up
cd ..
rm -rf ta-lib
rm ta-lib-0.4.0-src.tar.gz

```

and python requirements 

```bash

pip3 install -r requirements.txt
```


## Install every time when `src/` changes
```bash
./compile.sh
```


## feauture 

Use `ta-lib` and `pandas_ta` to calculate technical indicators, example data:

```text
time           2020-01-08 10:10:00 2020-01-08 10:15:00 2020-01-08 10:20:00 2020-01-08 10:25:00  ... 2020-01-08 10:40:00 2020-01-08 10:45:00 2020-01-08 10:50:00 2020-01-08 10:55:00
code                     sh.603369           sh.603369           sh.603369           sh.603369  ...           sh.603369           sh.603369           sh.603369           sh.603369
open                         87.46               87.38               87.27               88.15  ...               88.79               89.03               88.95               88.68
high                         87.65               87.38               88.28               88.47  ...               89.27               89.11               88.97               89.05                              low                          87.25               87.14               87.25               87.81  ...               88.79               88.81               88.68               88.60                              close                        87.38               87.27               88.10               88.36  ...               89.03               88.97               88.68               89.03
volume                      223312              139200              762410              508599  ...              554110              325797              156303              223714
amount                  7355007.00          4572699.00         25175884.00         16870310.00  ...         18564582.00         10910810.00          5224677.00          7479621.00
SMA_50                       86.61               86.63               86.67               86.72  ...               86.86               86.91               86.94               86.99
SMA_200                      86.15               86.16               86.18               86.20  ...               86.23               86.24               86.25               86.27
BBL_20_2.0                   86.20               86.17               85.99               85.85  ...               85.52               85.51               85.56               85.60
BBM_20_2.0                   86.73               86.75               86.82               86.91  ...               87.23               87.35               87.46               87.58
BBU_20_2.0                   87.27               87.33               87.64               87.96  ...               88.95               89.20               89.35               89.55
BBB_20_2.0                    1.24                1.34                1.90                2.44  ...                3.93                4.22                4.33                4.51
BBP_20_2.0                    1.10                0.95                1.27                1.19  ...                1.02                0.94                0.82                0.87
RSI_14                       64.68               61.72               72.29               74.72  ...               79.79               78.31               70.58               73.87
MACD_8_21_9                   0.17                0.20                0.32                0.44  ...                0.72                0.75                0.72                0.73
MACDh_8_21_9                  0.11                0.11                0.19                0.25  ...                0.30                0.27                0.19                0.16
MACDs_8_21_9                  0.06                0.09                0.13                0.19  ...                0.42                0.49                0.53                0.57
VOLUME_SMA_20            151884.40           153604.40           190144.90           214144.85  ...           274390.35           288020.20           291405.35           295791.05
feature_close                 0.00               -0.00                0.01                0.00  ...                0.00               -0.00               -0.00                0.00
feature_open                  1.00                1.00                0.99                1.00  ...                1.00                1.00                1.00                1.00
feature_high                  1.00                1.00                1.00                1.00  ...                1.00                1.00                1.00                1.00
feature_low                   1.00                1.00                0.99                0.99  ...                1.00                1.00                1.00                1.00
feature_volume               -0.35               -0.38                4.51               -0.33  ...                0.63               -0.41               -0.52                0.43

```
