FROM python:3.12
ADD ./src /src
COPY requirements.txt /
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
EXPOSE 8000