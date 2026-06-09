uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-grpc.txt
uv pip install -r requirements-alone.txt
uv pip install protobuf==4.25.3
