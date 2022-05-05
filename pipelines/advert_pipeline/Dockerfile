FROM tensorflow/tfx:1.7.1
WORKDIR /pipeline
COPY ./ ./
RUN pip install --upgrade pip && pip install python-dotenv
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"