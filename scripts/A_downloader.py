import baostock as bs
import pandas as pd
from tqdm import tqdm

lg = bs.login()
# rs = bs.query_hs300_stocks()
rs = bs.query_stock_industry()

stocks = []
while (rs.error_code == '0') & rs.next():
    stocks.append(rs.get_row_data())

for _, code, *args in tqdm(stocks):
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,amount,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
        start_date='2024-01-01', end_date='2024-12-31',
        frequency="d", adjustflag="1")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    result.to_csv(f"data/{code}.csv.gz", compression='gzip', index=False)

bs.logout()
