FROM python:3.7-stretch
WORKDIR /dashboard/
EXPOSE 8501
ENV TZ=Asia/Seoul
ENV VIRTUAL_ENV=/dashboard/.venv
ENV PATH=${VIRTUAL_ENV}/bin:$PATH
RUN python -m venv ${VIRTUAL_ENV}
COPY requirements.txt /dashboard/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /dashboard/logs
COPY . .
ENTRYPOINT ["streamlit", "run"]
CMD ["webapp.py"]
