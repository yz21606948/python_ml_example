import json
import boto3
import sagemaker
from sagemaker.s3 import S3Downloader
from sagemaker.processing import FrameworkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn

region = boto3.session.Session().region_name
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

preprocessing_job_description = script_processor.jobs[-1].describe()
output_config = preprocessing_job_description['ProcessingOutputConfig']
for output in output_config["Outputs"]:
	if output["OutputName"] == "train_data":
		preprocessed_training_data = output["S3Output"]["S3Uri"]
	if output["OutputName"] == "test_data":
		preprocessed_test_data = output["S3Output"]["S3Uri"]
print("preprocessed_training_data S3Uri is {}".format(preprocessed_training_data))
print("preprocessed_test_data S3Uri is {}".format(preprocessed_test_data))

sklearn = SKLearn(entry_point="train.py", 
				  source_dir="code",
				  framework_version="0.20.0", 
				  instance_type="ml.m5.xlarge",
				  role=role)
sklearn.fit({"train": preprocessed_training_data})
training_job_description = sklearn.jobs[-1].describe()
model_data_s3_uri = "{}{}/{}".format(
	training_job_description["OutputDataConfig"]["S3OutputPath"],
	training_job_description["TrainingJobName"],
	"output/model.tar.gz"
)
print("model_data_s3_uri is {}".format(model_data_s3_uri))

script_processor.run(
	code = "evaluation.py",
	source_dir='code',
	inputs = [
		ProcessingInput(source=model_data_s3_uri, destination="/opt/ml/processing/model"),
		ProcessingInput(source=preprocessed_test_data, destination='/opt/ml/processing/test'),
	],
	outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
)
evaluation_job_description = script_processor.jobs[-1].describe()
evaluation_output_config = evaluation_job_description['ProcessingOutputConfig']
for output in evaluation_output_config["Outputs"]:
	if output["OutputName"] == "evaluation":
		evaluation_s3_uri = output["S3Output"]["S3Uri"] + "/evaluation.json"
		break

evaluation_output = S3Downloader.read_file(evaluation_s3_uri)
evaluation_output_dict = json.loads(evaluation_output)
print(json.dumps(evaluation_output_dict, sort_keys=True, indent=4))

