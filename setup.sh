git -C emsdk/ pull || git clone https://github.com/emscripten-core/emsdk.git

cd emsdk
./emsdk install 2.0.8
./emsdk activate 2.0.8
source emsdk/emsdk_env.sh
source emsdk_env.sh

cd ..
wget https://github.com/Tencent/ncnn/releases/download/20220216/ncnn-20220216-webassembly.zip -nc
unzip ncnn-20220216-webassembly.zip -n

./build.sh