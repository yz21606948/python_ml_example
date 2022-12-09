import os
import boto3
import argparse
import pandas as pd
import awswrangler as wr

def _check_and_create_bucket(client, s3path):
	bucket = s3path.split("/")[2]
	all_buckets = client.list_buckets()['Buckets']
	is_exist = False
	if all_buckets:
		for item in all_buckets:
			if bucket == item['Name']:
				is_exist = True
	if not is_exist:
		print("{0} is not exist. It is creating ...")
		try:
			client.create_bucket(Bucket=bucket, 
								 CreateBucketConfiguration= {
								 	'LocationConstraint': 'cn-northwest-1'
								 })
		except Exception as e:
			print("Create {0} bucket failed: {1}".format(bucket, str(e)))
			raise e

def upload_parquet(s3path, filepath, mode):
	client = boto3.client('s3')
	print('#' * 30)
	print('Start upload {0} to {1} by {2} mode'.format(filepath, s3path, mode))
	_check_and_create_bucket(client=client, s3path=s3path)
	df = pd.read_csv(filepath)
	wr.s3.to_parquet(
		df=df,
		path=s3path,
		dataset=True,
		mode=mode
	)
	print("{0} bucket upload done".format(s3path))


def main():
	s3path = args.s3path
	filepath = args.filepath
	mode = args.mode
	upload_parquet(s3path=s3path, filepath=filepath, mode=mode)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Argument for S3 parquet sync")
	parser.add_argument("--s3path", help="S3 bucket path", required=True)
	parser.add_argument("--filepath", help="Local file path", required=True)
	parser.add_argument("--mode", 
						help="Support overwrite, appending", 
						choices={"overwrite", "appending"}, 
						default="overwrite")
	args = parser.parse_args()
	main()