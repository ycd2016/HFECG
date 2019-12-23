FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.1.0-cuda10.0-py3

MAINTAINER Chauncy Yao <cdyao@scutmsc.club>

ADD . /competition

WORKDIR /competition

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U; pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; pip install -r requirements.txt;

CMD ["sh", "run.sh"]
