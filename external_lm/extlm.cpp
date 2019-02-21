#include "external_lm.h"
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: extlm <path-to-exported-script-module> <path-to-vocab>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  ExternalLanguageModel lm(argv[1], argv[2]);
  
  std::vector<std::string> ex1;
  ex1.push_back("when");
  ex1.push_back("one");
  ex1.push_back("option");  
  ex1.push_back("is");
  ex1.push_back("selected");  
  std::cout<<"String: "<<ex1<<"   Score: "<<lm.score_phrase(ex1)<<std::endl;

  std::vector<std::string> ex2;
  ex2.push_back("where");  
  ex2.push_back("who");
  ex2.push_back("go");
  ex2.push_back("table");
  ex2.push_back("sit");  
  std::cout<<"String: "<<ex2<<"   Score: "<<lm.score_phrase(ex2)<<std::endl;

  std::vector<std::string> ex3;
  ex3.push_back("the");  
  ex3.push_back("fox");
  ex3.push_back("jumps");
  ex3.push_back("on");
  ex3.push_back("the");  
  ex3.push_back("box");  
  std::cout<<"String: "<<ex3<<"   Score: "<<lm.score_phrase(ex3)<<std::endl;
  

  std::vector<std::string> ex4;
  ex4.push_back("the");  
  ex4.push_back("fox");
  ex4.push_back("jump");
  ex4.push_back("at");
  ex4.push_back("the");  
  ex4.push_back("box");  
  std::cout<<"String: "<<ex4<<"   Score: "<<lm.score_phrase(ex4)<<std::endl;
}
