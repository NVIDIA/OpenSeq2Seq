#include <torch/script.h> // One-stop header
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class ExternalLanguageModel {
  private:
    std::shared_ptr<torch::jit::script::Module> module;
    std::unordered_map<std::string, long> token_to_id;

  public:
    ExternalLanguageModel(const char* path_to_traced_pytorch_model,
                          const char* path_to_vocabluary) {
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
    }

    float score_phrase(std::vector<std::string> words) {
      //iterate in order and get word indices
      assert(words.size()>0);
      std::vector<long> inpt;      
      for (std::vector<std::string>::size_type i=0; i!=words.size(); ++i) {
        inpt.push_back(this->token_to_id[words[i]]);
      }
      
      std::vector<torch::jit::IValue> inputs;      
      auto t = torch::from_blob(inpt.data(), {(long)inpt.size(), 1}, c10::TensorOptions(c10::ScalarType::Long));
      inputs.push_back(t);         
      
      auto output = module->forward(inputs);        
      float result = output.toTensor().item().toFloat();
      return result;
    };  
};





