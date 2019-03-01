#include "libtorch/include/torch/script.h" // One-stop header
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

class ExternalLanguageModel {
  private:
    std::shared_ptr<torch::jit::script::Module> module;
    std::unordered_map<std::string, long> token_to_id;

  public:
    ExternalLanguageModel(const char*);
    ExternalLanguageModel() {};
    float score_phrase(const std::vector<std::string>&);
};





