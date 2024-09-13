# Use the official Python 3.10 image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# COPY . /app
# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install -U "jax[cuda12]" && \
    pip install -r requirements.txt
    
# Clone the repositories
# You can add as many `git clone` commands as needed
RUN git clone https://github.com/christiando/dysts.git && \
    git clone https://github.com/christiando/darts.git && \
    git clone https://github.com/christiando/gaussian-toolbox.git && \
    git clone https://github.com/christiando/timeseries_models.git

# Change the working directory to the first cloned repository and install requirements
WORKDIR /app/gaussian-toolbox
RUN pip install .

# Repeat for the second repository if needed
WORKDIR /app/timeseries_models
RUN pip install .

# Set the default working directory back to /app
WORKDIR /app
RUN pip install holidays
RUN pip install pandas==1.3.5

# Add a script to trust all notebooks in /app/work
RUN echo '#!/bin/bash\njupyter trust /app/work/*.ipynb' > /app/trust_notebooks.sh && chmod +x /app/trust_notebooks.sh

# Create a new user with the same UID and GID as the host user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN addgroup --gid $GROUP_ID myuser && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID myuser

# Switch to the new user
USER myuser    
    
# Run Jupyter Lab by default (optional)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/app/work"]

