# Use official Python runtime
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash django
RUN chown -R django:django /app
USER django

# Expose port 8000
EXPOSE 8000

# Run Django migrations and start server
CMD ["sh", "-c", "python manage.py migrate && gunicorn --bind 0.0.0.0:${PORT:-8000} config.wsgi:application"]
