# 概述
`python_ml_docker` 是基于python的机器学习的容器化的一个例子

- `sklearn_example` 是基于python sklearn的示例

# 结构
结构由python脚本文件、 运行逻辑sh文件、 脚本运行环境Dockerfile构成。

## Python脚本
python脚本文件一般分为以下文件：
- 从云平台对象存储(S3)中读取文件
- 对Raw Data进行数据处理与特征工程
- 构建模型并训练模型
- 将以上产生的结果文件，调试日志文件，checkpoint文件上传到云平台(S3)中

## run.sh
`run.sh` 文件定义代码的执行逻辑。 一般顺序执行Python脚本即可， 特殊情况可以定义多线程并行执行Python脚本，例如需要并行批量处理数据

## Dockerfile 
`Dockerfile` 一般分为两层，一层为base环境, 一层为基于base环境的脚本执行环境
- Dockerfile_base
    - 定义最基础的运行环境，例如该环境应该安装Python3.10, 机器学习环境框架(sklearn, tensorlow, PyTorch), 云平台python SDK
    - 该image将上传到云平台的镜像仓库
- Dockerfile
    - 基于base运行环境，用户自定义环境。例如额外安装一些Python库，例如requests等。 例如以下代码表示基于腾讯云的sklearn环境。
    
    ```
    FROM ccr.ccs.tencentyun.com/dev_base/sklearn_qcloud:latest
    ```
    
    - 自定义环境变量供代码使用,例如
    
    ```
        os.environ.get('TENCENTCLOUD_SECRET_ID')
    ```
    
    - 执行`run.sh`

## 构建镜像
构建镜像可以执行以下命令:

- Build image。 可以使用"--build-arg"传入参数，例如传入云平台credentails。
    
```
docker build -f Dockerfile -t python_ml_docker:latest --build-arg SECRET_ID=<SECRET_ID> --build-arg SECRET_KEY=<SECRET_KEY> .
```  
    
- Run image。 可以理解为执行一个Job。

```
docker run -it python_ml_docker:latest /bin/sh
```

## 部署镜像(job)
部署镜像将指定集群节点类型（CPU节点/GPU节点）, 需要的资源大小(CPU、RAM), 优先级等。

<b>Note：<b>这部分尚未完成

