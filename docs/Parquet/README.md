## Parquet的概览

Apache Parquet是一种文件格式，旨在支持复杂数据的快速处理。
AWS 这样称道：**在Amazon S3中，Parquet与文本格式相比，加载速度快2倍，消耗的存储空间少6倍**

### Columnar结构

不同于row-based结构的文件，例如csv, Parquet 是column-based结构的文件。

**ROW-BASED 结构**

| name  | city      | score |
|-------|-----------|-------|
| Billy | BeiJing   | 90    |
| Jack  | ShangHai  | 80    |
| Lily  | GuangZhou | 100   |

**COLUMN-BASED 结构** 

| name  | Billy   | Jack     | Lily      |
|-------|---------|----------|-----------|
| city  | BeiJing | ShangHai | GuangZhou |
| score | 90      | 80       | 100       |

## Parquet的优势

在存储和分析大量数据时，Apache Parquet文件格式将会更具优势。下面是使用AWS Athena查询一个较大的公共数据集的结果对比：

|                          | CSV   | Parquet | Columns |
|--------------------------|-------|---------|---------|
| **Query time (seconds)** | 735   | 211     | 18      |
| **Data scanned (GB)**    | 372.2 | 10.29   | 18      |

如果是指定查询colunm的名字, Parquet还将指数级的提高查询效率和降低费用。

## 将csv文件转成Parquet并上传至S3

你可以运行示例代码 `S3_parquet_upload.py`, 运行命令可以这样：
```
python S3_parquet_upload.py --s3path s3://billytest/dataset/ --filepath test.csv
```

运行成功后，可以在Console上看到：

**值得注意的是：** 脚本可以添加`--mode` 参数。 mode参数只支持“overwrite”, "appending", 一个是覆盖远端文件，一个是追加到远端文件后，默认是overwrite

## 参考链接
1. [apache-parquet-why-use](https://www.upsolver.com/blog/apache-parquet-why-use)
2. [Parquet Datasets](https://aws-sdk-pandas.readthedocs.io/en/stable/tutorials/004%20-%20Parquet%20Datasets.html)

