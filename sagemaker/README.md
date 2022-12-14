## 背景

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) 是AWS数据分析和机器学习的服务。你可以使用Sagemaker来完成简单的数据预处理，训练模型流程。
你的脚本以及依赖都将加载到一个[Docker 容器](https://www.docker.com/resources/what-container)中运行。容器提供一个有效且独立的环境，保证运行时的一致性和训练的可靠性。

## 安装

```
pip3 install sagemaker-training
```

## 使用
SageMaker分为[Processing Job](https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/processing-job.html) 和 [Training Job](https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/train-model.html). Processing Job 主要负责数据的读取和预处理，Training Job 主要负责模型训练与评估。

**Processing Job:**
![image](https://user-images.githubusercontent.com/17400718/204239911-14e13579-c85f-4460-b32b-af9ac6911d38.png)

**Training Job**
![image](https://user-images.githubusercontent.com/17400718/204240073-13cc657e-479e-4100-8277-cf83b923bd13.png)


### 数据预处理
Processing Job将从S3读取数据，处理完数据后将把数据上传至S3。
![image](https://user-images.githubusercontent.com/17400718/204232069-0cf33793-65ee-4be5-8087-af5fd9500d84.png)

#### 实例化Processing Job类
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

#### 运行Processing Job
使用`FrameworkProcessor.run()`方法运行Processing Job. 

```
input_data = "s3://sagemaker-sample-data-{}/processing/census/census-income.csv".format(region)
script_processor.run(
	code='preprocessing.py',
	source_dir='code',
	inputs=[ProcessingInput(source=input_data, destination='/opt/ml/processing/input')],
	outputs=[
		ProcessingOutput(output_name="train_data", destination="s3://billy-ml/aws_sklearn_example/output/train_data", source="/opt/ml/processing/train"),
		ProcessingOutput(output_name="test_data", destination="s3://billy-ml/aws_sklearn_example/output/test_data", source="/opt/ml/processing/test"),
	],
	arguments=["--train-test-split-ratio", "0.2"],
)
```

`ProcessingInput`里面的`source`是数据在S3中的位置，`destination`是容器中脚本读取数据的位置，这里是`/opt/ml/processing/input`

`ProcessingOutput`里面的`source`是容器中脚本输出数据的位置，这里是`/opt/ml/processing`，`destination`是容器中输出到S3的位置。（Sagemaker SDK默认将会创建这个bucket，如果这里定义的bucket不存在

`arguments`是定义在`preprocessing.py`中的命令行参数，这里是`python preprocessing --train-test-split-ratio 0.2`

**值得注意的是：** 如果你需要引入其他依赖，你必须要`source_dir`中`requeirements.txt`文件里指定。Sagemaker SDK将在容器中自动安装。

##### 并行处理文件
当我们把`instance_count`设置为大于1时，需要将`s3_data_distribution_type="ShardedByS3Key"`设置到`ProcessingInput`中，才能实现并行处理文件。

![image](https://user-images.githubusercontent.com/17400718/204251724-af8927cf-97ba-4e1b-a261-f0d30b4aa011.png)

这时S3目标目录下的文件将会分成n份，Sagemaker将会平均分配到实例中处理这些文件。 你的脚本可以这样写：
```
for file in Path("/opt/ml/processing/input/").rglob('*.parquet'):
    file_path = str(file)
    # load file and process it.
```

### 训练模型
我们将创建`SKLearn`实例在training job中运行`train.py`
```
sklearn = SKLearn(entry_point="train.py", 
		  source_dir="code",
		  framework_version="0.20.0", 
		  instance_type="ml.m5.xlarge",
		  role=role)
sklearn.fit({"train": preprocessed_training_data})
```
这里同样需要定义执行training job的机器类型。

**值得注意的是:**`preprocessed_training_data`这个变量是数据在S3中的位置，这里是"3://billy-ml/aws_sklearn_example/output/train_data"。这个位置将默认映射到容器中“/opt/ml/input/data/train”. 所以`train.py`中读取数据应如下：
```
training_data_directory = '/opt/ml/input/data/train'
train_features_data = os.path.join(training_data_directory, "train_features.csv")
train_labels_data = os.path.join(training_data_directory, "train_labels.csv")
```

### 模型评估
`evaluation.py`是模型评估的脚本。我们将使用原先创建的实例`FrameworkProcessor`在processing job中运行它。

```
script_processor.run(
	code = "evaluation.py",
	source_dir='code',
	inputs = [
		ProcessingInput(source=model_data_s3_uri, destination="/opt/ml/processing/model"),
		ProcessingInput(source=preprocessed_test_data, destination='/opt/ml/processing/test'),
	],
	outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
)
```

### SageMaker Docker 容器中的目录结构
```
/opt/ml
├── input
│   ├── config
│   │   ├── hyperparameters.json
│   │   └── resourceConfig.json
│   └── data
│       └── <channel_name>
│           └── <input data>
├── model
│   └── <model files>
└── output
    └── failure
```
