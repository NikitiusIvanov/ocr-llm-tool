# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the Dash app will run on
ENV PORT 8080
EXPOSE $PORT

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Command to run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "main:server"]
