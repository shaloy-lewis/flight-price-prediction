FROM python:3.12.4-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file
COPY ./requirements.txt /app/requirements.txt

# Upgrade pip and install the dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the application files
COPY ./api.py /app/api.py
COPY ./init_setup.sh /app/init_setup.sh
COPY ./src /app/src
COPY ./artifacts/model.pkl /app/artifacts/model.pkl
COPY ./artifacts/preprocessor.pkl /app/artifacts/preprocessor.pkl
COPY ./artifacts/outlier_threshold.json /app/artifacts/outlier_threshold.json 

WORKDIR /app

# Set up the command to run the application
CMD ["sh", "init_setup.sh"]
