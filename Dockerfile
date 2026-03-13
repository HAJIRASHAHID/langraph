FROM python:3.11-slim      
# So slim is much smaller and faster to download.

 #sets working directory  inside container ======folder app
WORKDIR /app  
#All next commands run inside /app
# Step 1 — Install CPU-only torch FIRST (small = 200MB not 915MB!)
RUN pip install --no-cache-dir torch==2.2.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2 — Install everything else-----insisde app all files are copied and installed--------copy files from pc to container
COPY requirements.txt .

#no cache
#smaller image
RUN pip install --no-cache-dir -r requirements.txt

# Step 3 — Copy your code
#Copies your entire project folder into the container.
COPY . /app

#-p -----mean create folder if not exist
RUN mkdir -p downloads

ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

