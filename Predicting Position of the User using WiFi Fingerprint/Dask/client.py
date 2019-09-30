import pandas
import dask.dataframe as dd
from dask.distributed import Client

client = Client("10.110.122.238:8888")

df = pd.read_csv('trainingData.csv')
future = client.scatter(df)  # send dataframe to one worker
ddf = dd.from_delayed([future], meta=df)  # build dask.dataframe on remote data
ddf = ddf.repartition(npartitions=20).persist()  # split
client.rebalance(ddf)  # spread around all of your workers