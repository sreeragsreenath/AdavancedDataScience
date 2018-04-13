from distributed import Client, LocalCluster
cluster = LocalCluster()
client = Client(cluster)
def inc(x):
    return x + 1

def add(x, y):
    return x + y

a = client.submit(inc, 10)  # calls inc(10) in background thread or process
b = client.submit(inc, 20)
a.result()