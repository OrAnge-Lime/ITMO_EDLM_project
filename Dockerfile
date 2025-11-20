FROM ubuntu:22.04


RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY . /app
RUN test -d /app/weights || echo "Directory /app/weights does not exist. Check README"

RUN python3.11 -m venv /app/env


RUN /app/env/bin/pip install --upgrade pip && \
    /app/env/bin/pip install -r requirements.txt
RUN echo "source /app/env/bin/activate" >> /root/.bashrc


CMD ["/bin/bash"]
