import baostock as bs
import pandas as pd

lg = bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

rs = bs.query_history_k_data_plus(
    "sh.600000",
    "time,code,open,high,low,close,volume,amount",
    start_date='2020-01-01', end_date='2024-03-24',
    frequency="5", adjustflag="1")
print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)

result['time'] = pd.to_datetime(result['time'], format='%Y%m%d%H%M%S%f')
result['time'] = result['time'].dt.strftime('%Y-%m-%d %H:%M:%S')


result.to_csv("A_stock_5min.csv.gz", compression='gzip', index=False)
print(result)

bs.logout()
