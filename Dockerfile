FROM armswdev/tensorflow-arm-neoverse-n1:r21.12-tf-2.7.0-eigen
MAINTAINER Ryan

WORKDIR /darkchess-robot

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
RUN pip install jupyter

CMD jupyter notebook --ip='*' --port=8000 --no-browser --allow-root --notebook-dir="/"