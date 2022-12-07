import sagemaker
from time import gmtime, strftime
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

role = "arn:aws:iam::<your-account>:role/service-role/AmazonSageMakerServiceCatalogProductsExecutionRole"
# Download data from public dataset
session = sagemaker.Session()
s3_output = session.default_bucket()
s3_prefix = "R-in-Processing"
s3_source = "sagemaker-workshop-pdx"
session.download_data(path="./data", bucket=s3_source, key_prefix="R-in-Processing/us-500.csv")
# Upload data to your S3 bucket
rawdata_s3_prefix = "{}/data/raw".format(s3_prefix)
raw_s3 = session.upload_data(path="./data", key_prefix=rawdata_s3_prefix)
print(raw_s3)

# ScriptProcessor lets you run a command inside a Docker container.
script_processor = ScriptProcessor(
	command=["Rscript"],
	image_uri="<your-account>.dkr.ecr.us-west-2.amazonaws.com/billy.sagemaker/billy-r-insagemaker-processing",
	role=role,
	instance_count=1,
	instance_type="ml.m5.xlarge"
)

# Start the Sagemaker Processing job.
processing_job_name = "R-in-Processing-{}".format(strftime("%d-%H-%M-%S", gmtime()))
output_destination = "s3://{}/{}/data".format(s3_output, s3_prefix)

script_processor.run(
	code="preprocessing.R",
	job_name=processing_job_name,
	inputs=[ProcessingInput(source=raw_s3, destination="/opt/ml/processing/input")],
	outputs=[
		ProcessingOutput(
			output_name="csv",
			destination="{}/csv".format(output_destination),
			source="/opt/ml/processing/csv"
		),
		ProcessingOutput(
			output_name="images",
			destination="{}/images".format(output_destination),
			source="/opt/ml/processing/images"
		)
	]
)

# Retriving and viewing job results
preprocessing_job_description = script_processor.jobs[-1].describe()

output_config = preprocessing_job_description["ProcessingOutputConfig"]
for output in output_config["Outputs"]:
	if output["OutputName"] == "csv":
		preprocessing_csv_data = output["S3Output"]["S3Uri"]
	if output["OutputName"] == "images":
		preprocessing_images = output["S3Output"]["S3Uri"]

print("Preprocessing csv S3 uri is: {}".format(preprocessing_csv_data))
print("Preprocessing image data is: {}".format(preprocessing_images))
