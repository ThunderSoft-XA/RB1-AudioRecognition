cmake_minimum_required(VERSION 3.10)
project(Audio-recognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Set the path to the TensorFlow Lite library
set(TFLITE_LIB_PATH "/usr/local/lib/libtensorflow-lite.a")

# The path of the setting TensorFlow Lite header files
set(TFLITE_INCLUDE_PATH "/home/thundersoft/dev_C++/tensorflow/" "/home/thundersoft/dev_C++/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include/")

set(CMAKE_LIBRARY_PATH /usr/lib/aarch64-linux-gnu)

find_library(DL_LIB dl)
find_library(PTHREAD_LIB pthread)
find_library(LIBSNDFILE_LIBRARY sndfile)

# Add an executable
add_executable(Audio-recognition Audio-recognition.cpp)

# Link to TensorFlow Lite library
target_link_libraries(Audio-recognition "${TFLITE_LIB_PATH}")

target_link_libraries(Audio-recognition ${DL_LIB} ${PTHREAD_LIB})

target_link_libraries(Audio-recognition ${LIBSNDFILE_LIBRARY})

# Contains TensorFlow Lite header files
target_include_directories(Audio-recognition PRIVATE "${TFLITE_INCLUDE_PATH}")

