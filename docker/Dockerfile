# Imaxe base con Ubuntu 22.04 e CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Configurar variables de entorno
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    SETUPTOOLS_USE_DISTUTILS=stdlib \
    PATH="/opt/conda/bin:$PATH"

# Instalar dependencias do sistema
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    ca-certificates \
    bzip2 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Copiar environment.yml no directorio de traballo
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
COPY environment.yml /workspace/environment.yml
COPY vendor/Rex-Omni /workspace/vendor/Rex-Omni
WORKDIR /workspace

# Crear entorno Conda e activalo por defecto
RUN conda env create -f environment.yml -y
RUN /opt/conda/bin/conda init bash
RUN echo "conda activate tfg_lois_ssod-vocabulario-aberto" >> ~/.bashrc

# Configurar git para erros de permisos
RUN git config --global --add safe.directory /workspace

# Copiar entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Executar entrypoint script que personaliza o ambiente e inicia bash
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
