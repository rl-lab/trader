import baostock as bs
import pandas as pd
from tqdm import tqdm

lg = bs.login()
rs = bs.query_hs300_stocks()

hs300_stocks = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    hs300_stocks.append(rs.get_row_data())

for _, code, _ in tqdm(hs300_stocks):
    rs = bs.query_history_k_data_plus(
        code,
        "time,code,open,high,low,close,volume,amount",
        start_date='2020-01-01', end_date='2024-03-24',
        frequency="5", adjustflag="1")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    result['time'] = pd.to_datetime(result['time'], format='%Y%m%d%H%M%S%f')
    result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    result.to_csv(f"data/{code}.csv.gz", compression='gzip', index=False)

bs.logout()
