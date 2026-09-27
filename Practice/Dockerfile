FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir jupyter
RUN pip install --no-cache-dir seaborn

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

