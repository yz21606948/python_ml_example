# 同步本地目录数据到AWS S3 buckets指南

## 配置AWS CLI

1. 按照此[链接](https://aws.amazon.com/cli/) 下载以及安装AWS CLI。
2. 使用`aws configure` set up credentail.
	```
	aws configure
	AWS Access Key ID [None]: xxxxxxxxxxxxxxxxxx
	AWS Secret Access Key [None]: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
	Default region name [None]: cn-northwest-1
	Default output format [None]: json
	```
## 同步数据

### 创建S3 buckets

在创建S3 buckets时注意设置智能分层和过期数据处理。 在S3 bucket的生命周期管理中设置

### 智能分层 Amazon S3 Intelligent-Tiering

在S3 bucket的生命周期管理中设置

Amazon S3 Intelligent-Tiering (S3 Intelligent-Tiering) 是唯一一个在访问模式变化时通过在四个访问分层之间移动对象来自动节省成本的云存储类。S3 Intelligent-Tiering 存储类旨在通过自动将数据移动到最具成本效益的访问层来优化成本，而无操作开销。它通过将对象存储在四个访问层中来达到目的：两个低延迟访问层，针对频繁访问和不频繁访问进行了优化；以及两个可选存档访问层，专为异步访问而设计且针对罕见访问进行了优化。上传或转换至 S3 Intelligent-Tiering 的对象自动存储在频繁访问分层中。为了实现每个对象每月只需少量的监视和自动化费用，Amazon S3 会在 S3 Intelligent-Tiering 中监控对象的访问模式，然后将连续 30 天内未访问的对象移动到不频繁访问分层。您可以激活一个或两个存档访问层，可以将 90 天未访问的对象自动移动到存档访问层，然后在 180 天后将其移动到深度存档访问层。如果对象后来被访问，S3 Intelligent-Tiering 会将对象移回到频繁访问分层。这意味着，存储在 S3 Intelligent-Tiering 中的所有对象在需要时始终可用。当使用 S3 Intelligent-Tiering 存储类时，没有检索费用，当在访问层之间移动对象时，也没有额外的分层费用。S3 Intelligent-Tiering 没有最短存储持续时间。S3 Intelligent-Tiering 设计的最小对象大小为 128KB，用于自动分层。您可以将这些对象存储在 S3 Intelligent-Tiering 中，但它们不会被监控，并且将会始终按频繁访问层费率收费，没有监控和自动化费用。对于那些访问模式未知或不可预测的数据，它是理想的存储类。

### 过期处理

在S3 bucket的生命周期管理中设置

可以参考一下例子：

1. 把所有的生命管理规则应用到所有objects.
2. 勾选“lifecycle rule actions”的对应设置

这里设置，当对象上传后7天自动移到Intelligent-Tiering。数据过期时间为90天，即90天后过期数据自动清除。（这个可以另外创建一个life management rule 对部分数据进行设置。）


### 同步数据

本地目录是source源，S3 bucket是destination源，按照以下命令去同步。

```
aws s3 sync s3://<bucket-path> <local-path>
```

S3 bucket是source源，本地目录是destination源，按照以下命令去同步。

```
aws s3 sync <local-path> s3://<bucket-path>
```

## 定时任务

我们应该设置一个定时任务去定时执行同步数据的脚本，保证本地和远端的数据尽量一直。（注意这里应该是本地数据同步到远端，因为S3上传数据不需要钱，下在数据到本地 ¥1/GB）

## 数据使用

如果是本地调试，我们应该使用本地数据。如果是使用sagemaker job 或者 EC2执行任务，应该使用S3数据。原因[aws s3定价](https://www.amazonaws.cn/s3/pricing/)

|                             | 定价（宁夏）  |
|-----------------------------|---------------|
| **数据传入至 Amazon S3**    |               |
| 所有传入数据                | 每 GB ¥0.000  |
| **数据自 Amazon S3 传出至** |               |
| 同一区域的 Amazon EC2       | 每 GB ¥0.000  |
| 数据传出至互联网            | 每 GB ¥ 0.933 |

