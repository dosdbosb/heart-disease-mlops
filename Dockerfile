# Start from an official, lightweight Python image (not a full OS —
# keeps the final image small and the build fast)
FROM python:3.12-slim

# All commands from here on happen inside /app in the container
WORKDIR /app

# Copy just the dependency list first (not all code yet) — Docker caches
# this step, so if only your code changes later (not requirements),
# rebuilding skips re-installing everything, saving time
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual code and saved model into the container
COPY api/ ./api/
COPY models/ ./models/

# Tell Docker this container listens on port 8000
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]