# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/tools/gen_op_registration.h>
#include <sndfile.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <regex>
#include <filesystem>
#include <string>
#include <math.h>

// Target sampling rate
const int SAMPLE_RATE = 16000;

// A comparison function to sort elements in descending order of size
bool compareElements(const std::pair<float, int>& a, const std::pair<float, int>& b) {
    return a.first > b.first;
}

//Get the index of the probability of the top 10
std::vector<int> getTopTenIndices(const std::vector<float>& inputVector) {
    std::vector<std::pair<float, int>> indexedVector;
    // Create a new vector with an index
    for (int i = 0; i < inputVector.size(); ++i) {
        indexedVector.push_back(std::make_pair(inputVector[i], i));
    }
    // Copy the top ten elements into the new vector
    std::vector<std::pair<float, int>> topTenElements(10);
    std::partial_sort_copy(indexedVector.begin(), indexedVector.end(), topTenElements.begin(), topTenElements.end(),
                        compareElements);

    // The elements in the new vector are sorted to get the indices of the top ten elements
    std::sort(topTenElements.begin(), topTenElements.end(), compareElements);

    std::vector<int> topTenIndices;
    for (const auto& element : topTenElements) {
        topTenIndices.push_back(element.second);
    }
    return topTenIndices;
}

//All label information in the csv file is extracted
std::vector<std::string> class_names(std::string class_map_csv) {
    std::ifstream csv_file(class_map_csv);
    std::vector<std::string> class_names;

    if (csv_file.is_open()) {
        std::string line;
        std::getline(csv_file, line); // Skip file headers
        while (std::getline(csv_file, line)) {
            std::istringstream iss(line);
            std::string num,label;
            std::getline(iss, label, ',');
            std::getline(iss, num);
            class_names.push_back(label);
        }
        csv_file.close();
    }
    return class_names;
}

//Sample rate resampling
std::vector<float> resample(const std::vector<float>& waveform, float sourceRate, float targetRate) {
    // The scaling factor for resampling is calculated using the desired sampling rate
    float resampleFactor = targetRate / sourceRate;
    // Calculate the length of the output waveform
    int outputLength = std::ceil(waveform.size() * resampleFactor);
    // Create the output waveform vector
    std::vector<float> resampledWaveform(outputLength);
    // Do resampling
    for (int i = 0; i < outputLength; i++) {
        float sourceIndex = i / resampleFactor;
        int leftIndex = std::floor(sourceIndex);
        int rightIndex = std::min(leftIndex + 1, static_cast<int>(waveform.size()) - 1);
        float fraction = sourceIndex - leftIndex;

        // The resampled values are computed using linear interpolation
        resampledWaveform[i] = (1 - fraction) * waveform[leftIndex] + fraction * waveform[rightIndex];
    }
    return resampledWaveform;
}

int main()
{
  // Load the tflite model
  const char* model_path = "../cfg/yamnet.tflite";
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
  // Create the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  // Distribution tensor
  interpreter->AllocateTensors();
  // Get input and output details
  std::vector<int> inputs = interpreter->inputs();
  std::vector<int> outputs = interpreter->outputs();

  //Read csv files
  std::vector<std::string> yamnet_classes = class_names("../cfg/yamnet_class_map.csv");

  //Get the filenames of all files in wav format in the folder
  std::string audios_path = "../audios";
  // Define the filename pattern to match: files ending in.wav
  std::regex filePattern(".+\\.wav$");
  std::vector<std::string> audio_list;
  // Iterate over all the files in the folder
  for (const auto& entry : std::filesystem::directory_iterator(audios_path)) {
      std::string audioname = entry.path().filename().string();

      // Use regular expressions to match filenames
      if (std::regex_match(audioname, filePattern)) {
          audio_list.push_back(audioname); // Prints the matched filenames
      }
  }

  for(std::string au_name:audio_list)
  {
    // Read audio file data, sample rate, number of channels
    std::vector<int16_t> wav_data;
    int sr;
    int channels;
    const char* audio_path = "../audios/";
    std::string originalString(audio_path); //Convert char* to std::string
    std::string audio_append = audio_path + au_name;
    const char* audio_allname = audio_append.c_str();
    SF_INFO sf_info;
    SNDFILE* sf = sf_open(audio_allname, SFM_READ, &sf_info);
    if (sf == nullptr) {
      std::cout<<audio_allname<<"Open False !"<<std::endl;
      return 0;
    }
    wav_data.resize(sf_info.frames);
    sf_readf_short(sf, wav_data.data(), sf_info.frames);
    sr = sf_info.samplerate;
    channels = sf_info.channels;

    // The number of sampling points for each channel is calculated
    sf_count_t samples = sf_info.frames;
    sf_close(sf);

    //Converts audio data range to [-1.0, +1.0]
    std::vector<float> waveform(wav_data.size());
    for(size_t i = 0; i < wav_data.size(); i++){
      waveform[i] = wav_data[i] / 32768.0;
    }

    //For multi-channel audio, the average value is processed into a single channel
    if(channels > 1){
      for (sf_count_t i = 0; i < samples; ++i) {
          sf_readf_float(sf, waveform.data(), 1); // Read multi-channel data

          float sum = 0.0;
          for (int j = 0; j < channels; ++j) {
              sum += waveform[j];
          }
          float average = sum / channels;

          waveform[i] = average; // Store single-channel data
      }
    }
    //If the sampling rate and the predetermined sampling rate are not consistent, resample
    if(sr != SAMPLE_RATE){
      waveform = resample(waveform, sr, SAMPLE_RATE);
    }

    // Get the input tensor
    int input_index = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_index);

    // The waveform is converted to the shape and type of the input tensor
    float* input_data = interpreter->typed_tensor<float>(input_index);
    const int input_size = input_tensor->bytes / sizeof(float);
    for (int i = 0; i < input_size; ++i) {
        input_data[i] = waveform[i];
    }

    //Inference run
    interpreter->Invoke();

    // Get the pointer and size of the output Tensor
    int output_index = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_index);
    float* output_data = interpreter->typed_tensor<float>(output_index);
    const int output_size = output_tensor->bytes / sizeof(float);
    std::vector<float> scores(output_size);
    for(int i=0; i < output_size; ++i){
      scores[i] = output_data[i];
    }

    // Calculating the average
    std::vector<float> prediction(interpreter->tensor(output_index)->dims->data[1]);
    for (int i = 0; i < prediction.size(); i++) {
        float sum = 0.0;
        for (int j = 0; j < interpreter->tensor(output_index)->dims->data[0]; j++) {
            sum += output_data[j * prediction.size() + i];
        }
        prediction[i] = sum / interpreter->tensor(output_index)->dims->data[0];
    }

    // Get the first 10 indices
    std::vector<int> top10_i =  getTopTenIndices(prediction);;

    // Get the labels and probabilities
    std::vector<std::string> labels(10);
    std::vector<float> probabilities(10);
    for (int i = 0; i < 10; i++) {
        labels[i] = yamnet_classes[top10_i[i]];
        probabilities[i] = prediction[top10_i[i]];
    }
    //Remove the " from all elements within all labels.
    for (std::string& label : labels) {
        label.erase(std::remove(label.begin(), label.end(), '\"'), label.end());
    }

    // Print the results
    std::cout << audio_allname << ":" << std::endl;
    std::cout << "ids: { ";
    for (int i = 0; i < 10; i++) {
        if(i!= 9){
          std::cout << top10_i[i] << ", ";
        }
        else{
          std::cout << top10_i[i] << "}"<<std::endl;
        }
    }
    std::cout << "names: { " ;
    for(int i = 0; i < 10; i++)
    {
      if(i!=9){
        std::cout<<labels[i]<<", ";
      }
      else{
        std::cout<<labels[i];
      }
    }
    std::cout << "}" << std::endl;
    std::cout << "scores: { " << std::fixed << std::setprecision(3) << probabilities[0];
    for (int i = 1; i < 10; i++) {
        std::cout << ", " << probabilities[i];
    }
    std::cout << "}" << std::endl;
    std::cout<<std::endl;
  }
  return 0;
}
