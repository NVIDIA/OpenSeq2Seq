/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2018 Mozilla Corporation
 *
 * The code was taken from Mozilla DeepSpeech project:
 * https://github.com/mozilla/DeepSpeech/tree/master/native_client
 *
 */

#ifndef ALPHABET_H
#define ALPHABET_H

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

/*
 * Loads a text file describing a mapping of labels to strings, one string per
 * line. This is used by the decoder, client and Python scripts to convert the
 * output of the decoder to a human-readable string and vice-versa.
 */
class Alphabet {
public:
  Alphabet(const char *config_file) {
    std::ifstream in(config_file, std::ios::in);
    unsigned int label = 0;
    for (std::string line; std::getline(in, line);) {
      if (line.size() == 2 && line[0] == '\\' && line[1] == '#') {
        line = '#';
      } else if (line[0] == '#') {
        continue;
      }
      label_to_str_[label] = line;
      str_to_label_[line] = label;
      ++label;
    }
    size_ = label;
    in.close();
  }

  const std::string& StringFromLabel(unsigned int label) const {
    assert(label < size_);
    auto it = label_to_str_.find(label);
    if (it != label_to_str_.end()) {
      return it->second;
    } else {
      // unreachable due to assert above
      abort();
    }
  }

  unsigned int LabelFromString(const std::string& string) const {
    auto it = str_to_label_.find(string);
    if (it != str_to_label_.end()) {
      return it->second;
    } else {
      std::cerr << "Invalid label " << string << std::endl;
      abort();
    }
  }

  size_t GetSize() {
    return size_;
  }

  bool IsSpace(unsigned int label) const {
    //TODO: we should probably do something more i18n-aware here
    const std::string& str = StringFromLabel(label);
    return str.size() == 1 && str[0] == ' ';
  }

private:
  size_t size_;
  std::unordered_map<unsigned int, std::string> label_to_str_;
  std::unordered_map<std::string, unsigned int> str_to_label_;
};

#endif //ALPHABET_H
