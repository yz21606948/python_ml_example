## 背景

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) 是AWS数据分析和机器学习的服务。你可以使用Sagemaker来完成简单的数据预处理，训练模型流程。
你的脚本以及依赖都将加载到一个[Docker 容器](https://www.docker.com/resources/what-container)中运行。容器提供一个有效且独立的环境，保证运行时的一致性和训练的可靠性。

## 安装

```
pip3 install sagemaker-training
```

## 使用
SageMaker分为[Processing Job](https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/processing-job.html) 和 [Training Job](https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/train-model.html). Processing Job 主要负责数据的读取和预处理，Training Job 主要负责模型训练与评估。

### Processing Job
Processing Job将从S3读取数据，处理完数据后将把数据上传至S3。
![image](https://user-images.githubusercontent.com/17400718/204232069-0cf33793-65ee-4be5-8087-af5fd9500d84.png)

创建一个`FrameworkProcessor`，它将使用自定义框架让你的脚本运行在Processing Job中。
```
import sagemaker
from sagemaker.processing import FrameworkProcessor

role = "arn:aws:iam::<your_account>:role/service-role/AmazonSageMakerServiceCatalogProductsExecutionRole"
est_cls = sagemaker.sklearn.estimator.SKLearn
framework_version_str = "0.20.0"

script_processor = FrameworkProcessor(
	role = role,
	instance_count = 1,
	instance_type = "ml.m5.xlarge",
	estimator_cls = est_cls,
	framework_version = framework_version_str
)
```
这里可以指定运行job的机器类型`instance_type`、机器数量`instance_count`以及基础框架`estimator_cls`
对于机器类型，可以参考这个[链接](https://aws.amazon.com/cn/sagemaker/pricing/)
对于基础python框架，可以参考这个[链接](https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/how-it-works-training.html)目前支持：**Apache Spark**, **scikit-learn**, **Hugging Face**, **MXNet**, **PyTorch**, **TensorFlow**, **XGBoost**

使用`FrameworkProcessor.run()`方法运行
