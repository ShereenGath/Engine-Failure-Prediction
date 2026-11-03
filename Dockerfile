# 1. Start from a small official Linux image that already has Python 3.12
FROM python:3.12-slim

# 2. XGBoost needs this system library on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. All following commands run inside /app in the container
WORKDIR /app

# 4. Install only the libraries the API needs (copied first so Docker can cache this step)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 5. Copy the code and the trained model
COPY src/ src/
COPY models/ models/

# 6. The API listens on port 8000
EXPOSE 8000

# 7. Command that runs when the container starts
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
