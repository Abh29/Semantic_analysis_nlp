#include "ft_threads.h"
#include "tokenizer.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/tensor.h>
#include <algorithm>
#include <c10/core/DeviceType.h>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>
#include <strings.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/cuda.h>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>
#include <unordered_set>
#include <utility>

using json = nlohmann::json;
constexpr int threadCount = 11;
constexpr int64_t tokenCount = 512;
constexpr int64_t batch_size = 64;
FullTokenizer *pTokenizers[threadCount] = {};
std::string stop_words;
std::unordered_set<size_t> stop_ids;
auto options = torch::TensorOptions().dtype(torch::kInt64);

struct Tokenized {
  std::vector<std::wstring> tokens;
  std::vector<int64_t> ids;
  bool set = false;
  int label = 0;
};

using batch_ids = std::vector<std::vector<int64_t>>;

std::vector<Tokenized> train_dataset(25001);
std::vector<Tokenized> test_dataset(25001);

std::vector<torch::Tensor> train_tensors;
std::vector<torch::Tensor> test_tensors;

void normalizeIds(std::vector<int64_t> &ids) {
  // remove stopwords,
  ids.erase(std::remove_if(
                ids.begin(), ids.end(),
                [](size_t id) { return stop_ids.find(id) != stop_ids.end(); }),
            ids.end());
  ids.resize(tokenCount, 0);
}

std::string &toLowerCase(std::string &str) {
  // move to lowercase and remove punctuation
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  str.erase(std::remove_if(str.begin(), str.end(), ispunct), str.end());
  return str;
}

void work(std::string &line, int index, std::vector<Tokenized> &dataset) {
  // read the text tokenize, normalize and save the result
  json data;
  data = json::parse(line);
  auto text = data["text"].dump();
  text = toLowerCase(text);
  auto tokens = pTokenizers[index % threadCount]->tokenize(text);
  auto ids = pTokenizers[index % threadCount]->convertTokensToIds(tokens);
  normalizeIds(ids);
  dataset[index].tokens = std::move(tokens);
  dataset[index].ids = std::move(ids);
  dataset[index].set = true;
  dataset[index].label = data["label"];
}

void readStopWords(const std::string &path) {
  auto tokenizer =
      std::make_unique<FullTokenizer>("bert-base-uncased-vocab.txt");
  std::ifstream f(path);
  std::getline(f, stop_words);
  f.close();

  auto tokens = tokenizer->tokenize(stop_words);
  auto ids = tokenizer->convertTokensToIds(tokens);
  stop_ids.insert(ids.begin(), ids.end());
}

void prepareDataSet(const std::string &path, ft::ThreadPool::pointer &tp,
                    std::vector<Tokenized> &dataset) {
  std::ifstream stream(path);
  std::string line;
  json data;
  int i = 0;
  while (std::getline(stream, line)) {
    tp->addTask(work, line, i++, dataset);
  }
  std::cout << "reached end of file " << std::endl;

  stream.close();
}

void createTensors(std::vector<torch::Tensor> &tensors,
                   std::vector<Tokenized> &dataset) {

  std::vector<int64_t> v(tokenCount, 0);
  auto options = torch::TensorOptions().dtype(torch::kInt64);
  int i = 0;
  while (i < dataset.size()) {
    if (!dataset[i].set && ++i)
      continue;
    torch::Tensor batch_tensor =
        torch::empty({batch_size, tokenCount}, options);
    int j = 0;
    while (j < batch_size) {
      if (i < dataset.size() && dataset[i].set) {
        batch_tensor[j] = torch::tensor(dataset[i++].ids, options);
      } else {
        batch_tensor[j] = torch::tensor(v, options);
      }
      ++j;
    }
    tensors.push_back(batch_tensor);
  }
}

void checkData(std::vector<Tokenized> &dataset) {
  for (auto &s : dataset)
    if (s.set && s.ids.size() != tokenCount)
      std::cout << "error size != 512" << std::endl;
}

struct StraitForwardModel : public torch::nn::Module {
  using pointer = std::shared_ptr<StraitForwardModel>;

  StraitForwardModel(int64_t input_size, int64_t hidden_size,
                     int64_t output_size) {
    // Define the first fully connected layer
    _fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
    // Define the second fully connected layer
    _fc2 = register_module("fc2", torch::nn::Linear(hidden_size, output_size));
  }
  // Implement the forward pass
  torch::Tensor forward(torch::Tensor x) {
    // Apply the first layer and a ReLU activation
    x = torch::relu(_fc1->forward(x));
    // Apply the second layer
    x = _fc2->forward(x);
    // Apply a sigmoid activation to get the output probability
    return torch::sigmoid(x);
  }

private:
  torch::nn::Linear _fc1{nullptr}, _fc2{nullptr};
};

int main(int argc, char **argv) {
  if (argc < 3)
    return 1;

  try {
    for (int i = 0; i < threadCount; ++i) {
      pTokenizers[i] = new FullTokenizer("bert-base-uncased-vocab.txt");
    }
  } catch (std::exception &e) {
    std::cerr << "construct FullTokenizer failed" << std::endl;
    return -1;
  }

  ft::ThreadPool::pointer threadPool =
      std::make_shared<ft::ThreadPool>(threadCount);

  // prepare the datasets for training and testing
  try {
    readStopWords("english_stop_words.txt");
    prepareDataSet(argv[1], threadPool, train_dataset);
    prepareDataSet(argv[2], threadPool, test_dataset);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }

  // joint all threads
  threadPool.reset();

  checkData(train_dataset);
  checkData(test_dataset);

  // create the model
  StraitForwardModel::pointer model =
      std::make_unique<StraitForwardModel>(512, 64, 1);

  // create the tensors
  createTensors(train_tensors, train_dataset);
  createTensors(test_tensors, test_dataset);

  for (int i = 0; i < threadCount; ++i)
    delete pTokenizers[i];
  return 0;
}
