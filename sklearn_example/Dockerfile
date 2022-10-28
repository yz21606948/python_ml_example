FROM ccr.ccs.tencentyun.com/dev_base/sklearn_qcloud:latest

ARG SECRET_ID
ARG SECRET_KEY
WORKDIR /workdir

ENV TENCENTCLOUD_SECRET_ID=$SECRET_ID
ENV TENCENTCLOUD_SECRET_KEY=$SECRET_KEY

COPY * /workdir/

ENTRYPOINT ["/bin/sh", "/workdir/run.sh"