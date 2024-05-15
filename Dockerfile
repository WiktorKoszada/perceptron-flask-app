FROM python:3.9-slim
WORKDIR /app
COPY app.py perceptron.py model.pkl requirements.txt /app/
COPY model_train.py /app/
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]