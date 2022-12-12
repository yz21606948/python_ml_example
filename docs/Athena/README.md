## Athena 概览

Amazon Athena 是一种交互式查询服务，让您可以轻松使用标准 SQL 语言来分析 Amazon S3 中的数据。Athena 属于无服务器服务，因此不需要管理基础设施，且您仅需为您运行的查询付费。

## Athena 定价

使用 Amazon Athena，您只需为您运行的查询付费。将根据每个查询扫描的数据量向您收费。您可以通过压缩、分区或将数据转换为列式格式实现显著的成本节省和性能提升，因为每一项操作都会减少 Athena 需要扫描以执行查询的数据量。

### 每次查询的价格

中国（宁夏）：¥34.34 / TB-扫描

中国（北京）：¥41.20 / TB-扫描

若使用Parquet数据格式，将节约大量成本。 具体可以参考这个[链接](https://www.amazonaws.cn/athena/pricing/)

## 使用Athena加载数据

你可以运行`python Athena_loaddata.py`来运行示例代码，去加载`s3://billytest/dataset`中的`students`表的数据。

### awswrangler调用Athena

awswrangler有三种方法使用Athena查询然后获取到dataframe。具体用法可以参考这个[链接](https://aws-sdk-pandas.readthedocs.io/en/stable/tutorials/006%20-%20Amazon%20Athena.html)

- ctas_approach=True（Default）
- unload_approach=True and ctas_approach=False
- ctas_approach=False

执行完脚本查询后，查询结果也会在S3中备份下来。

**值得注意的是以下能更省费用且提高效率：** 

1. 我们在查询中最好指定需要查询的列名，即：`SELECT city, name, score FROM students` 而不是 `SELECT * FROM students`
2. 我们存在S3中的数据最好是Parquet数据结构，这样费用能省60%-90%