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
