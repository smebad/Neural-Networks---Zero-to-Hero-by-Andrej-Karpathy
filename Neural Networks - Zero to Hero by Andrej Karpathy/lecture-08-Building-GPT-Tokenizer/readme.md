# Lecture 8: Building GPT Tokenization from Scratch

## Introduction
In this lecture, I explored the process of tokenization as used in GPT-style language models. Tokenization is the method of converting text into a sequence of tokens (usually integers) that can be processed by a neural network. This lecture deepened my understanding of encoding and decoding by covering the following:

- How to work with Unicode strings and various encodings (UTF-8, UTF-16, UTF-32).
- How to manually tokenize text by converting it to bytes and then to integer tokens.
- How to implement a simple Byte Pair Encoding (BPE)-like algorithm to merge frequent token pairs and compress the vocabulary.
- How to define functions for encoding text into token IDs and decoding token IDs back into text.
- How to use regular expressions to mimic GPT-2 tokenization patterns.
- An introduction to tokenization libraries such as **tiktoken** and **SentencePiece**, including training a SentencePiece model on a toy dataset.

These techniques are fundamental for preparing text data for language models and provide insight into the inner workings of modern tokenizers.

## What I Learned
- **Unicode and Encodings:**  
  I learned how Python represents strings in Unicode and how to convert these strings into bytes using different encodings (UTF-8, UTF-16, UTF-32). This is crucial for handling text in multiple languages and symbol sets.

- **Manual Tokenization and BPE:**  
  By converting text to bytes and then to a list of integers, I created a raw token sequence. I implemented functions to compute the frequency of adjacent token pairs (`get_stats`) and to merge the most frequent pairs (`merge`). This iterative merging process is similar to Byte Pair Encoding (BPE), which is used to build compact vocabularies for models like GPT.

- **Encoding and Decoding:**  
  I developed `encode()` and `decode()` functions that convert text to token IDs and back to text. This deepened my understanding of how models “read” and “generate” text.

- **Regex Tokenization:**  
  I experimented with regular expressions to emulate tokenization rules similar to those used in GPT-2. This helped in understanding how complex tokenization patterns can be designed.

- **Tokenization Libraries:**  
  I explored the **tiktoken** library, which implements GPT-2 and GPT-4 style tokenization, and the **SentencePiece** library, a popular tool for subword tokenization. I even trained a SentencePiece model on a toy text file to see how these tokenizers are built and used in practice.

## Code Overview
1. **Unicode and Byte-Level Encoding:**  
   - Demonstrated how to work with Unicode strings and convert them to bytes.
   - Converted a given text to a list of integers representing raw byte tokens.

2. **Manual Token Merging (BPE-like Algorithm):**  
   - Implemented `get_stats()` to count adjacent token pair frequencies.
   - Implemented `merge()` to replace frequent token pairs with a new token.
   - Iteratively merged token pairs to achieve a compressed vocabulary size.
   - Built a vocabulary mapping from token IDs to byte sequences.
   - Developed `encode()` and `decode()` functions for tokenizing and detokenizing text.

3. **Regular Expression Tokenization:**  
   - Used regex patterns to split text similarly to GPT-2’s tokenizer.
   - Tested the regex on sample strings to verify its behavior.

4. **Using tiktoken and SentencePiece:**  
   - Demonstrated how to use **tiktoken** for encoding and decoding with GPT-2 and GPT-4 tokenizers.
   - Trained a SentencePiece model on a toy dataset and used it to encode/decode text, showing how subword tokenization can be applied.

5. **Loading Pre-trained Tokenizers:**  
   - Loaded GPT-2 style encoder JSON and BPE merges from files to illustrate how tokenization pipelines are structured in real-world models.

## Conclusion
In Lecture 8, I expanded my knowledge of tokenization—a critical step in building effective language models like GPT. I learned how to:
- Handle Unicode and various text encodings.
- Manually implement a BPE-like token merging algorithm to compress vocabulary.
- Encode text into token IDs and decode token IDs back into human-readable text.
- Utilize modern tokenization libraries such as **tiktoken** and **SentencePiece** to streamline the tokenization process.

These insights deepen my understanding of how language models process and generate text, and they provide a strong foundation for future work in natural language processing. I look forward to applying these techniques in more advanced projects and further exploring the nuances of text encoding and tokenization.
