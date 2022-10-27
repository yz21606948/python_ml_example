ARG SECRET_ID
ARG SECRET_KEY

FROM ccr.ccs.tencentyun.com/dev_base/sklearn_qcloud:latest

WORKDIR /workdir

ENV TENCENTCLOUD_SECRET_ID $SECRET_ID 
ENV TENCENTCLOUD_SECRET_KEY $SECRET_KEY 

COPY *.py /workdir/

CMD ["/usr/local/bin/python","/workdir/download.py"]
CMD ["/usr/local/bin/python","/workdir/train.py"]
CMD ["/usr/local/bin/python","/workdir/upload.py"]