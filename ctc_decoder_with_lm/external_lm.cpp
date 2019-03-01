#include "external_lm.h"
#include <exception>

ExternalLanguageModel::ExternalLanguageModel(const char* path_to_nm_prefix) {
      try {
          std::string path_to_traced_pytorch_model = std::string(path_to_nm_prefix) + std::string(".pt");
          std::string path_to_vocabluary = std::string(path_to_nm_prefix) + std::string(".voc");
          this->module = torch::jit::load(path_to_traced_pytorch_model);
          std::cout<<"Done loading model"<<std::endl;

          std::string word;
          long word_id;

          std::ifstream infile(path_to_vocabluary);
          long count = 0;
          while (infile >> word >> word_id) {
            this->token_to_id.insert(std::make_pair(word, word_id));
            count += 1;
          }
          std::cout<<"Done loading vocabluary "<<"Loaded: "<<count<<" tokens."<<std::endl;
      } catch (...) {
        std::cout<<">>>>>>>>>>>>> FAILED TO INITIALIZE NEURAL LANGUAGE MODEL"<<std::endl;
      }
    };

float ExternalLanguageModel::score_phrase(const std::vector<std::string> &words) {
      //iterate in order and get word indices
      std::vector<long> inpt;
      for (std::vector<std::string>::size_type i=0; i!=words.size(); ++i) {
        inpt.push_back(this->token_to_id[words[i]]);
      }
      
      std::vector<torch::jit::IValue> inputs;      
      auto t = torch::from_blob(inpt.data(), {(long)inpt.size(), 1}, at::TensorOptions(at::ScalarType::Long));
      inputs.push_back(t);         
      
      auto output = module->forward(inputs);        
      float result = output.toTensor().item().toFloat();
      return result;
    };  





