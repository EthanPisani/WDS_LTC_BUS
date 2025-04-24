FROM python:3.12-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
# Copy app.y, Model.py, best_model.pth, feature_scaler.pkl, target_scaler.pkl, stop_id_mapping.json
COPY app.py .
COPY Model.py .
COPY best_model.pth .
COPY feature_scaler.pkl .
COPY target_scaler.pkl .
COPY stop_id_mapping.json .


# Expose the port
EXPOSE 5240

# Set the entrypoint
CMD ["python", "app.py"]