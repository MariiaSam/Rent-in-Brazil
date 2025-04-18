
# Use the official Python 3.12 image as a base image
FROM python:3.12

# set the working directory
WORKDIR /app

# install Poetry
RUN pip install poetry

# Copy the project configuration file and dependency file to the container
COPY pyproject.toml poetry.lock ./

# Setting dependencies through poetry
RUN pip poetry install

RUN poetry install --no-root


# Copy the entire project to the container
COPY . .

# Expose the port that the application listens on.
EXPOSE 8880

# Specify the default command to run main.py via poetry
CMD ["poetry", "streamlit", "run", "python", "app.py"]