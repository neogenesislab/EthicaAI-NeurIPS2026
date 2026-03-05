FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Default: run full reproduction (论文 수치 재현)
# For quick smoke test, override with: docker run ... python reproduce_quick.py --fast
CMD ["python", "reproduce_quick.py"]
