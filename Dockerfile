FROM --platform=linux/amd64 python:3.11-slim AS example-nntest
# import basic python container to ensure pytorch installed with GPU drivers via PIP below

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
#ENV PYTHONUNBUFFERED=1

RUN pip install  --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install  --user nnunetv2

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

RUN python -m pip install --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

COPY --chown=user:user DEEP_PSMA_Infer.py /opt/app/
COPY --chown=user:user deep_psma_utils.py /opt/app/
COPY --chown=user:user nnunet_config_paths.py /opt/app/


ENTRYPOINT ["python", "inference.py"]
#ENTRYPOINT ["/bin/bash"] #switch entry point to enable interactive session for testing "-it" flag


