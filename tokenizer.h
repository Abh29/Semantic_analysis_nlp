/* c++ version of tokenization for bert
   Copyright (C) 2019  luistung
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.*/

#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utf8proc.h>
#include <vector>

// https://unicode.org/reports/tr15/#Norm_Forms
// https://ssl.icu-project.org/apiref/icu4c/uchar_8h.html

const std::wstring stripChar = L" \t\n\r\v\f";
using Vocab = std::unordered_map<std::wstring, size_t>;
using InvVocab = std::unordered_map<size_t, std::wstring>;

class BasicTokenizer {
public:
  BasicTokenizer(bool doLowerCase = true);
  std::vector<std::wstring> tokenize(const std::string &text) const;

private:
  std::wstring cleanText(const std::wstring &text) const;
  bool isControol(const wchar_t &ch) const;
  bool isWhitespace(const wchar_t &ch) const;
  bool isPunctuation(const wchar_t &ch) const;
  bool isChineseChar(const wchar_t &ch) const;
  std::wstring tokenizeChineseChars(const std::wstring &text) const;
  bool isStripChar(const wchar_t &ch) const;
  std::wstring strip(const std::wstring &text) const;
  std::vector<std::wstring> split(const std::wstring &text) const;
  std::wstring runStripAccents(const std::wstring &text) const;
  std::vector<std::wstring> runSplitOnPunc(const std::wstring &text) const;

  bool mDoLowerCase;
};

class WordpieceTokenizer {
public:
  WordpieceTokenizer(std::shared_ptr<Vocab> vocab,
                     const std::wstring &unkToken = L"[UNK]",
                     size_t maxInputCharsPerWord = 200);
  std::vector<std::wstring> tokenize(const std::wstring &text) const;

private:
  std::shared_ptr<Vocab> mVocab;
  std::wstring mUnkToken;
  size_t mMaxInputCharsPerWord;
};

class FullTokenizer {
public:
  FullTokenizer(const std::string &vocabFile, bool doLowerCase = true);
  std::vector<std::wstring> tokenize(const std::string &text) const;
  std::vector<std::wstring> tokenizeLength(const std::string &text,
                                           uint32_t length) const;
  std::vector<int64_t>
  convertTokensToIds(const std::vector<std::wstring> &text) const;

private:
  std::shared_ptr<Vocab> mVocab;
  InvVocab mInvVocab;
  std::string mVocabFile;
  bool mDoLowerCase;
  BasicTokenizer mBasicTokenizer;
  WordpieceTokenizer mWordpieceTokenizer;
};

static std::string normalize_nfd(const std::string &s) {
  std::string ret;
  char *result = (char *)utf8proc_NFD((unsigned char *)s.c_str());
  if (result) {
    ret = std::string(result);
    free(result);
    result = NULL;
  }
  return ret;
}

static bool isStripChar(const wchar_t &ch) {
  return stripChar.find(ch) != std::wstring::npos;
}

static std::wstring strip(const std::wstring &text) {
  std::wstring ret = text;
  if (ret.empty())
    return ret;
  size_t pos = 0;
  while (pos < ret.size() && isStripChar(ret[pos]))
    pos++;
  if (pos != 0)
    ret = ret.substr(pos, ret.size() - pos);
  pos = ret.size() - 1;
  while (pos != (size_t)-1 && isStripChar(ret[pos]))
    pos--;
  return ret.substr(0, pos + 1);
}

static std::vector<std::wstring> split(const std::wstring &text) {
  std::vector<std::wstring> result;
  boost::split(result, text, boost::is_any_of(stripChar));
  return result;
}

static std::vector<std::wstring> whitespaceTokenize(const std::wstring &text) {
  std::wstring rtext = strip(text);
  if (rtext.empty())
    return std::vector<std::wstring>();
  return split(text);
}

static std::wstring convertToUnicode(const std::string &text) {
  size_t i = 0;
  std::wstring ret;
  while (i < text.size()) {
    wchar_t codepoint;
    utf8proc_ssize_t forward =
        utf8proc_iterate((utf8proc_uint8_t *)&text[i], text.size() - i,
                         (utf8proc_int32_t *)&codepoint);
    if (forward < 0)
      return L"";
    ret += codepoint;
    i += forward;
  }
  return ret;
}

static std::string convertFromUnicode(const std::wstring &wText) {
  char dst[64];
  std::string ret;
  for (auto ch : wText) {
    utf8proc_ssize_t num = utf8proc_encode_char(ch, (utf8proc_uint8_t *)dst);
    if (num <= 0)
      return "";
    ret += std::string(dst, dst + num);
  }
  return ret;
}

static std::wstring tolower(const std::wstring &s) {
  std::wstring ret(s.size(), L' ');
  for (size_t i = 0; i < s.size(); i++) {
    ret[i] = utf8proc_tolower(s[i]);
  }
  return ret;
}

static std::shared_ptr<Vocab> loadVocab(const std::string &vocabFile) {
  std::shared_ptr<Vocab> vocab(new Vocab);
  size_t index = 0;
  std::ifstream ifs(vocabFile, std::ifstream::in);
  if (!ifs) {
    throw std::runtime_error("open file failed");
  }
  std::string line;
  while (getline(ifs, line)) {
    std::wstring token = convertToUnicode(line);
    if (token.empty())
      break;
    token = strip(token);
    (*vocab)[token] = index;
    index++;
  }
  return vocab;
}
