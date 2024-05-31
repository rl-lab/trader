import baostock as bs
import pandas as pd
from tqdm import tqdm

lg = bs.login()
# rs = bs.query_hs300_stocks()
rs = bs.query_stock_industry()

stocks = []
while (rs.error_code == '0') & rs.next():
    stocks.append(rs.get_row_data())

dayq = "date,code,open,high,low,close,amount,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
minq = "date,time,code,open,high,low,close,amount"

for _, code, *args in tqdm(stocks):
    rs = bs.query_history_k_data_plus(
        code, minq,
        start_date='2000-01-01', end_date='2024-12-31',
        frequency="30", adjustflag="1")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result['time'] = pd.to_datetime(result['time'], format='%Y%m%d%H%M%S%f')
    result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result = result.round(3)

    result.to_csv(f"data/{code}.30min.csv.gz", compression='gzip', index=False)

bs.logout()
