### 从腾讯云COS S3上读取数据
# 1. 设置用户配置，包括secretId, secretKey 以及 Region
import os
import sys
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys

secret_id = os.environ.get('TENCENTCLOUD_SECRET_ID')
secret_key = os.environ.get('TENCENTCLOUD_SECRET_KEY')
region = 'ap-guangzhou'
config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
# 2. 获取客户端对象
client = CosS3Client(config)

# 3. 上传
# 对于较小文件，直接下载
response = client.upload_file(
    Bucket='titanic-1302638621',
    Key='data/submission.csv',
    LocalFilePath='submission.csv',
    EnableMD5=False,
    progress_callback=None
)