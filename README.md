# ncnn-webassembly-yolov5

open https://nihui.github.io/ncnn-webassembly-yolov5 and enjoy

# quick build and deploy

1. Allow execution for the shell script files

```
chmod +x ./setup.sh
chmod +x ./build.sh
```

2. Run `./setup.sh`

3. Run the web server and access http://localhost:8000 from the browser
`python3 -m http.server --directory deploy`

# build and deploy

1. Install emscripten
```shell
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 2.0.8
./emsdk activate 2.0.8

source emsdk/emsdk_env.sh
```

2. Download and extract ncnn webassembly package
```shell
wget https://github.com/Tencent/ncnn/releases/download/20220216/ncnn-20220216-webassembly.zip
unzip ncnn-20220216-webassembly.zip
```

3. Build four WASM feature variants
```shell
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DWASM_FEATURE=basic ..
make -j4
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DWASM_FEATURE=simd ..
make -j4
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DWASM_FEATURE=threads ..
make -j4
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DWASM_FEATURE=simd-threads ..
make -j4
```

4. Deploy the *.data *.js *.wasm and *.html files to your web server
```
# deploy files
deploy/
├── index.html
├── yolov5-basic.data
├── yolov5-basic.js
├── yolov5-basic.wasm
├── yolov5-simd.data
├── yolov5-simd.js
├── yolov5-simd-threads.data
├── yolov5-simd-threads.js
├── yolov5-simd-threads.wasm
├── yolov5-simd-threads.worker.js
├── yolov5-simd.wasm
├── yolov5-threads.data
├── yolov5-threads.js
├── yolov5-threads.wasm
├── yolov5-threads.worker.js
└── wasmFeatureDetect.js
```
5. Deploy local server(python3 as a example)
```
python3 -m http.server --directory deploy
```
https://gist.github.com/nihui/c97a27b302bc56d554874af0f158d6df

6. Access local server(chrome as a example)
```
# launch chrome browser, enter following command to address bar and press ENTER: 
chrome://flags/#unsafely-treat-insecure-origin-as-secure

# enter following keyword to "Search flags" and press ENTER:
"insecure origins"
you will find "Insecure origins treated as secure" key

#enter local server url and click right side dropdown list, select "Enabled"
url example: http://192.168.1.100:8000

#relaunch chrome browser and access http://192.168.1.100:8000 (replace 192.168.1.100 with your local ip)
```
