FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

ADD requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ADD . .
# Make port 5000 available to the world outside this container
EXPOSE 5000

RUN python build_model.py

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run app.py when the container launches
CMD ["flask", "--app", "model_api", "run"]
