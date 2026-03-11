FROM python:3.11-slim

WORKDIR /app

# Step 1 — Install CPU-only torch FIRST (small = 200MB not 915MB!)
RUN pip install --no-cache-dir torch==2.2.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2 — Install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 3 — Copy your code
COPY . /app

RUN mkdir -p downloads

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]