---
layout: post
title:  "bert tokenizer源码解析"
date:   2023-01-05 09:20:08 +0800
category: "AI"
published: true
---
做序列标注时，label和token之间的对应关系至关重要。但是大多数tokenizer都会对原始的字符序列做一定的修改，这对保持token和label之间的对应关系造成了一定的影响。因此，有必要对tokenizer的细节行为有一个清楚的认识。本文以bert tokenzier为例说明里面的细节。
<!--more-->

# 一、tokenize框架
bert的tokenizer分为两步，首先是BasicTokenizer，然后是WordPieceTokenizer。从FullTokenizer的源码可以看出来：

```python
class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    #先后调用basic_tokenizer和wordpiece_tokenizer完成tokenize
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)
```
下面分别研究BasicTokenizer和WordPieceTokenizer的源码。

# 二、BasicTokenizer
看看BasicTokenizer，它首先做了文本的标准化和清洗，确保后面的处理保持一致，不出错。

## 1、文本的标准化和清洗-引起文本变化的第一处
分别对应convert_to_unicode和_clean_text两个函数。
下面看看convert_to_unicode函数
```python
def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")
```
上面的函数综合考虑了python 2 和 3 两个版本。并默认输入的编码是utf-8。

再看看_clean_text函数：
```python
  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)

      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)
```
这里就涉及到了字符的变化。首先code point是0和0xfffd的会被丢弃。0表示控制字符null terminator, 0xfffd表示unicode无法识别的字符。
其次，传统的控制字符也会被丢弃。
但是，看一下_is_control的源码：
```python
def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False
```
会发现，\t \n \r没有被算作控制字符，而是作为空白符处理了。我想这是因为这三种字符特别常见，也有一定的语义分割的作用。出于保留尽可能完整的上下文的目的，转化成空格。

unicode类别中Cc Cf表示  control和format，属于控制排版的字符。

同时，所有的whitespace， 会统一替换为空格。这里看一下_is_whitespace的源码：
```python
def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False
```
可以看到上面提到的\t \n \r都认为是空格了。同时unicode类别中。Zs表示Separator, Space的意思。所以转化成空格了。

由此可以看到，clean_text是可能引起字符串长度的变化的。

## 2、中文的tokenize
为了支持中文字符，完成文本标准化和清洗后，紧接着就是调用函数_tokenize_chinese_chars。
我们看看源码：
```python
  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False
```
逻辑相对清晰，就是在一个中文字符前后添加空格，这样后续就不用考虑中文英文了，因为英文本身就是空格分隔的。

## 3、空格tokenize
紧接着，就是用空格分隔出基本都token。
```python
def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens
```

这里其实解决了之前的一个疑问，中文tokenzie时，在字符前后都添加了空格，势必会造成两个汉字之间有两个空格。这里就能看出来，这没什么影响。
至此，就得到了一个初始的token序列了。

## 4、标准化、标点分割--引起文本变化的第二处
得到初始的token后，紧接着对token做了两步处理，首先是去掉字符上下的声调小字符。如Ç下面的那个小尾巴。方法就是将该字符用分解形式表示。
先看代码：
```python
  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)
```
代码中，NFD表示normal form decomposition，将那些可以用组合的单字符表示，也可以用分解的字符序列表示的字符，统一用分解的形式。
然后再把分解后的小尾巴去掉，这样，就确保了token的长度是不变的。但是字符却是发生了变化的。去掉小尾巴的操作就是代码中Mn的continue语句。
Mn表示的是Nonspacing Mark， 具体包含的字符可以在这里查看：
https://www.compart.com/en/unicode/category/Mn

可以发现基本就是小尾巴字符。

做完这一步后，对每个token进一步按照标点符号进行分割。代码如下：
```python
  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]
```
这里需要注意的是_is_punctuation这个方法，它的判断标准和我们初始直觉是不一样的。

```python
def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False
```
除了传统意义上的标点符号，所有非字母数字的ASCII字符都看做标点符号。这需要注意一下。

至此，BasicTokenizer的功能就完成了。
其完整代码如下：
```python
class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.
    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)
```

# 三、WordPieceTokenzier
这部分完全是为了英文单词的处理。中文其实不需要的。
核心就是将一个单词分解为sub word，这样做的目的可以有效缓解OOV问题。

代码如下：
```python
class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.
    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.
    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]
    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.
    Returns:
      A list of wordpiece tokens.
    """
    #再次执行一次转成unicode，绝不信任调用方
    text = convert_to_unicode(text)

    output_tokens = []
    # 再次执行一次空格分隔，绝不信任调用方
    for token in whitespace_tokenize(text):
      chars = list(token)
      # 如果一个单词的长度超过了最大的字符限制，输出位unk
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        #这个循环是从最后往前移，目的是找到从start开始能匹配到的最长的subword
        #找到后，将start 更新为end值。
        while start < end:
          substr = "".join(chars[start:end])
          #非第一个sub str，前面添加##用来标识。
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        # 没有找到subword，说明该单词不认识，用is_bad来标记，然后输出为unk
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens

```
