mkdir -p build >/dev/null 2>&1
cd build

cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DWASM_FEATURE=basic ..
make -j4
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DWASM_FEATURE=simd ..
make -j4
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DWASM_FEATURE=threads ..
make -j4
cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake -DWASM_FEATURE=simd-threads ..
make -j4

cd ..

find ./ -wholename './build/*.data' -exec cp -prv '{}' './deploy' ';'
find ./ -wholename './build/*.wasm' -exec cp -prv '{}' './deploy' ';'
find ./ -wholename './build/*.js' -exec cp -prv '{}' './deploy' ';'
find ./ -wholename './assets/*.jpg' -exec cp -prv '{}' './deploy' ';'
cp ./wasmFeatureDetect.js ./deploy/wasmFeatureDetect.js
cp ./index.html ./deploy/index.html