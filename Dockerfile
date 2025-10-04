# Use official Python runtime
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files into container
COPY . .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash django
RUN chown -R django:django /app
USER django

# Expose port 8000 (Render will map it automatically)
EXPOSE 8000

# Run migrations and start the app
CMD ["sh", "-c", "python manage.py migrate --noinput --skip-checks || echo 'Migrations failed, but continuing...' && gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000} --timeout 120 --log-level debug"]
