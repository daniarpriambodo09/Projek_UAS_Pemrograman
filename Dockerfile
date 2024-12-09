FROM apache/airflow:2.10.2

# Install system dependencies
USER root
RUN apt-get update && \
    apt-get install -y python3-pip libpq-dev && \
    apt-get clean

# Install Python packages
RUN pip install --upgrade pip
RUN pip install newspaper3k nltk requests pandas beautifulsoup4 lxml[html_clean]

# Switch back to airflow user
USER airflow

# Copy DAGs to Airflow
COPY dags/ /opt/airflow/dags/

# Command to start Airflow
CMD ["bash", "-c", "airflow db init && airflow webserver & airflow scheduler"]
