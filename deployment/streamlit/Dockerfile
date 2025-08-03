FROM python:3.11-slim

WORKDIR /app

# Copy Streamlit app
COPY deployment/streamlit/app.py .

# Optional: if you have a separate requirements file for Streamlit
COPY deployment/streamlit/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]