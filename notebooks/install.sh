# set CMAKE_ARGS=-DLLAMA_CUBLAS=on
# set FORCE_CMAKE=1
# pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

pip install flash-attn==2.5.8 --no-build-isolation --no-cache-dir
pip install -r requirements.txt
