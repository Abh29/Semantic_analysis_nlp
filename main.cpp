#include "ft_threads.h"
#include "tokenizer.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/tensor.h>
#include <algorithm>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <ostream>
#include <random>
#include <string>
#include <strings.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <unordered_set>
#include <utility>

using json = nlohmann::json;
constexpr int threadCount = 11;
constexpr int64_t tokenCount = 256;
constexpr int64_t batch_size = 64;
constexpr int epochs = 3;
FullTokenizer *pTokenizers[threadCount] = {};
std::string stop_words;
std::unordered_set<size_t> stop_ids;
auto options = torch::TensorOptions().dtype(torch::kFloat);

struct Tokenized {
  std::vector<std::wstring> tokens;
  std::vector<int64_t> ids;
  bool set = false;
  int64_t label = 0;
};

using batch_ids = std::vector<std::vector<int64_t>>;

std::vector<Tokenized> train_dataset;
std::vector<Tokenized> test_dataset;

std::vector<torch::Tensor> train_tensors;
std::vector<torch::Tensor> train_labels_tensors;
std::vector<torch::Tensor> test_tensors;
std::vector<torch::Tensor> test_labels_tensors;

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

void work(std::string line, int index, std::vector<Tokenized> *dataset) {
  // read the text tokenize, normalize and save the result
  json data;
  data = json::parse(line);
  auto text = data["text"].dump();
  text = toLowerCase(text);
  auto tokens = pTokenizers[index % threadCount]->tokenize(text);
  auto ids = pTokenizers[index % threadCount]->convertTokensToIds(tokens);
  normalizeIds(ids);
  (*dataset)[index].tokens = std::move(tokens);
  (*dataset)[index].ids = std::move(ids);
  (*dataset)[index].set = true;
  (*dataset)[index].label = data["label"];
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
  std::cout << "end read stop words" << std::endl;
}

void initialize_vector(const std::string &path, std::vector<Tokenized> &v) {
  int i = 0;
  std::string line;
  std::ifstream f(path);
  while (std::getline(f, line))
    ++i;
  f.close();
  v.resize(i);
}

void prepareDataSet(const std::string &path, ft::ThreadPool::pointer &tp,
                    std::vector<Tokenized> *dataset) {

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

void createTensors(std::vector<torch::Tensor> &data_tensors,
                   std::vector<torch::Tensor> &label_tensors,
                   std::vector<Tokenized> &dataset) {

  std::cout << "creating tensors from dataset size: " << dataset.size()
            << std::endl;
  std::vector<int64_t> v(tokenCount, 0);
  int i = 0;
  while (i < dataset.size()) {
    if (!dataset[i].set) {
      std::cout << "continue i: " << i << "\n";
      ++i;
      continue;
    }
    torch::Tensor batch_tensor =
        torch::empty({batch_size, tokenCount}, options);
    torch::Tensor batch_labels = torch::empty({batch_size, 1}, options);
    int j = 0;
    while (j < batch_size) {
      if (i < dataset.size() && dataset[i].set) {
        batch_tensor[j] = torch::tensor(dataset[i].ids, options);
        batch_labels[j] = torch::tensor(dataset[i].label, options);
      } else {
        batch_tensor[j] = torch::tensor(v, options);
        batch_labels[j] = torch::tensor(0.0, options);
      }
      ++j;
      ++i;
    }
    data_tensors.push_back(batch_tensor);
    label_tensors.push_back(batch_labels);
  }
  std::cout << "end create tensors" << std::endl;
}

struct StraitForwardModel : public torch::nn::Module {
  using pointer = std::shared_ptr<StraitForwardModel>;

  StraitForwardModel(int64_t input_size, int64_t hidden_size1,
                     int64_t hidden_size2, int64_t output_size) {
    // First fully connected layer
    _fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size1));
    // Second fully connected layer
    _fc2 =
        register_module("fc2", torch::nn::Linear(hidden_size1, hidden_size2));
    // Output layer
    _fc3 = register_module("fc3", torch::nn::Linear(hidden_size2, output_size));
  }
  // Implement the forward pass
  torch::Tensor forward(torch::Tensor x) {
    // Apply the first layer with a ReLU activation function
    x = torch::relu(_fc1->forward(x));
    // Apply the second layer with a ReLU activation function
    x = torch::relu(_fc2->forward(x));
    // Apply the output layer; no activation here
    x = _fc3->forward(x);
    return x;
  }

private:
  torch::nn::Linear _fc1{nullptr}, _fc2{nullptr}, _fc3{nullptr};
};

void shuffle_dataset(std::vector<torch::Tensor> &dataset,
                     std::vector<torch::Tensor> &labels) {

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  auto engine = std::default_random_engine(seed);
  int rand_pos = 0;
  int size = std::min(dataset.size(), labels.size());

  for (int i = 0; i < size; ++i) {
    rand_pos = std::rand() % size;
    std::swap(dataset[i], dataset[rand_pos]);
    std::swap(labels[i], labels[rand_pos]);
  }
};

void train(StraitForwardModel::pointer &model) {

  std::cout << "begin training " << std::endl;
  // Define the Binary Cross Entropy loss
  //  auto criterion = torch::nn::BCELoss();
  auto criterion = torch::nn::BCEWithLogitsLoss();
  // Define the optimizer, passing the network's parameters and specifying the
  // learning rate
  torch::optim::SGD optimizer(model->parameters(),
                              torch::optim::SGDOptions(0.0001));
  // iterate over the batches
  for (int i = 0; i < train_tensors.size(); ++i) {
    // Zero the gradients
    optimizer.zero_grad();
    // Forward pass: Compute predicted output by passing inputs to the model
    // train_tensors[i].to(torch::kFloat);
    auto output = model->forward(train_tensors[i]);
    // Calculate the loss
    auto loss = criterion(output, train_labels_tensors[i]);
    // Backward pass: Compute gradient of the loss with respect to model
    // parameters
    loss.backward();
    // Perform a single optimization step (parameter update)
    optimizer.step();
    if (i % 10 == 0) {
      std::cout << "Loss after " << i << " iterations: " << loss.item<double>()
                << std::endl;
    }
  }
}

// TODO: this is as good as a random guesser
// need to add a word embedding layer for the model
// in order for it to make any sense
int main(int argc, char **argv) {
  if (argc < 3)
    return 1;

  std::cout << "GPU "
            << (torch::cuda::is_available() ? "detected" : "not detected")
            << std::endl;

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
    initialize_vector(argv[1], train_dataset);
    prepareDataSet(argv[1], threadPool, &train_dataset);
    initialize_vector(argv[2], test_dataset);
    prepareDataSet(argv[2], threadPool, &test_dataset);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }

  // joint all threads
  threadPool.reset();

  // create the model
  StraitForwardModel::pointer model =
      std::make_unique<StraitForwardModel>(tokenCount, 128, 32, 1);

  // suffle the training data
  std::shuffle(train_dataset.begin(), train_dataset.end(),
               std::default_random_engine(31));

  // create the tensors
  createTensors(train_tensors, train_labels_tensors, train_dataset);
  createTensors(test_tensors, test_labels_tensors, train_dataset);

  // int k = 0;
  // for (auto &t : train_tensors) {
  //   std::cout << t << std::endl;
  //   ++k;
  //   if (k > 10)
  //     break;
  // }
  // std::cout << std::endl;

  // train the model
  for (int i = 0; i < epochs; ++i) {
    shuffle_dataset(train_tensors, train_labels_tensors);
    train(model);
  }

  std::cout << "endl! " << std::endl;
  // getchar();
  for (int i = 0; i < threadCount; ++i)
    delete pTokenizers[i];
  return 0;
}
