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
time            2020-01-08 10:10:00  2020-01-08 10:15:00  2020-01-08 10:20:00  2020-01-08 10:25:00  ...  2020-01-08 10:40:00  2020-01-08 10:45:00  2020-01-08 10:50:00  2020-01-08 10:55:00
open                           1.01                 1.01                 1.01                 1.02  ...                 1.03                 1.03                 1.03                 1.03
high                           1.01                 1.01                 1.02                 1.02  ...                 1.03                 1.03                 1.03                 1.03
low                            1.05                 1.05                 1.05                 1.06  ...                 1.07                 1.07                 1.07                 1.07
close                          1.04                 1.03                 1.04                 1.05  ...                 1.05                 1.05                 1.05                 1.05
volume                    223312.00            139200.00            762410.00            508599.00  ...            554110.00            325797.00            156303.00            223714.00
amount                   7355007.00           4572699.00          25175884.00          16870310.00  ...          18564582.00          10910810.00           5224677.00           7479621.00
feature_close                  0.00                -0.00                 0.01                 0.00  ...                 0.00                -0.00                -0.00                 0.00
feature_open                   0.98                 0.98                 0.97                 0.97  ...                 0.97                 0.98                 0.98                 0.97
feature_high                   0.98                 0.98                 0.98                 0.98  ...                 0.98                 0.98                 0.98                 0.98
feature_low                    1.02                 1.02                 1.01                 1.01  ...                 1.02                 1.02                 1.02                 1.01
feature_volume                -0.35                -0.38                 4.51                -0.33  ...                 0.63                -0.41                -0.52                 0.43
SMA_50                         1.03                 1.03                 1.03                 1.03  ...                 1.03                 1.03                 1.03                 1.03
SMA_200                        1.02                 1.02                 1.02                 1.02  ...                 1.02                 1.02                 1.02                 1.02
BB_UPPER                       1.03                 1.03                 1.04                 1.04  ...                 1.05                 1.06                 1.06                 1.06
BB_MIDDLE                      1.03                 1.03                 1.03                 1.03  ...                 1.03                 1.03                 1.04                 1.04
BB_LOWER                       1.02                 1.02                 1.02                 1.02  ...                 1.01                 1.01                 1.01                 1.01
RSI_14                        64.68                61.72                72.29                74.72  ...                79.79                78.31                70.58                73.87
MACD                           0.00                 0.00                 0.00                 0.01  ...                 0.01                 0.01                 0.01                 0.01
MACD_SIGNAL                    0.00                 0.00                 0.00                 0.00  ...                 0.00                 0.01                 0.01                 0.01
EMA_20                         1.03                 1.03                 1.03                 1.03  ...                 1.04                 1.04                 1.04                 1.04
ATR_14                         0.02                 0.02                 0.02                 0.02  ...                 0.02                 0.02                 0.02                 0.02
CCI_20                       263.50               173.03               254.79               235.61  ...               175.66               141.89               108.38               101.69
Aroon_Osc                     50.00                50.00                64.29                71.43  ...                92.86                92.86                64.29                64.29
```
