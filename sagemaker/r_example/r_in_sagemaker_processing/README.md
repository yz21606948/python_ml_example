## 在Sagemaker中使用R processing job

在这个例子中，我们将使用R sagemaker processing job, 它涉及一下步骤：

- 编写一个R脚本
- 创建docker 容器
- 创建一个sagemaker processing job
- 获取结果

### R 脚本

你可以在你本地R Studio 进行调试

### 创建Docker容器

在这个例子中， Dockerfile在docker目录下，Dockerfile不必包含你的R 脚本因为Sagemaker processing job 将会自动注入它。（这使得你能更灵活的修改你的脚本，你不用每次重新build docker image 如果你需要修改你的脚本）

以下命令我们将可以通过sagemaker访问我们编写的镜像。( **在此之前，你需要配置通过 `aws config` 配置aws crendentail** )

- Build local image
- Create ECR repository
- Push the image to ECR

```
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account>.dkr.ecr.<your-region>.amazonaws.com

aws ecr create-repository --region <your-region> --repository-name billy.sagemaker/billy-r-insagemaker-processing

docker build -f dockerfile -t billy.sagemaker/billy-r-insagemaker-processing:latest .

docker tag billy.sagemaker/billy-r-insagemaker-processing:latest <your-account>.dkr.ecr.<your-region>.amazonaws.com/billy.sagemaker/billy-r-insagemaker-processing:latest

docker push <your-account>.dkr.ecr.<your-region>.amazonaws.com/billy.sagemaker/billy-r-insagemaker-processing
```

### 创建sagemaker processing job & 获取结果

执行 `build.py` 即可。 可以参考主页的描述来设置 `ScriptProcessor` 的参数


