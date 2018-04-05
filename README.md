# LightLS
LightLS is a simple, language-agnostic tool for lexical text simplification

## Description

LightLS performs simple rule-based lexical simplification, primarily based on:

1. Estimates of word complexity based on their frequency in a large corpus;
2. Estimates of semantic similarities between words based on comparison of their vectors (i.e., embeddings) in a distributional space. 

The decision whether some word will be substituted with a "simpler" word, depends on several criteria: 

- The complexity of the target word from the input text
- The complexity (simplicity) of candidate replacement words
- The similarity of a candidate replacement word and the target word from the input text
- The similarity of a candidate replacement word with words in the context of the target word from the input text

## Running the tool

To run the tool, you need to run the script *simplifier.py*. The script requires four mandatory arguments and has additional optional arguments that can be used to regulate the rate of text simplification. To see the full list of arguments (mandatory and optional), run the script with the *-h* option, i.e., *python simplifier.py -h*.

### Mandatory arguments

The following are the mandatory arguments for the tool (to be provided in the order listed below): 

1. Path to the directory containing the files with texts to be simplified
2. Path to the output directory, where simplified texts and files listing performed lexical substitutions will be stored
3. Path to the file containing the word frequencies, pre-computed on a large corpus (we provide example files for English and Italian in the *resources* directory)
4. Path to the file containing pre-trained word embeddings (embeddings file in textual format is required, .bin and Gensim formats are not supported)

### Optional arguments

Optional arguments serve to fine-tune the lexical simplification for the types of texts you are looking to simplify (i.e., to adjust them for a new domain, language, etc.). Using these parameters you can control the amount/rate of simplification -- i.e., the tool can be configured to perform simplifications liberally or conservatively. The following are the optional arguments: 

1. \-s (or \-\-stopwords): a path to the file containing the list of stopwords for the language. Stopwords are never considered for simplification, nor can any other word be replaced by a stopword. If there are some words you never want to be simplified, you can simply add them to the stopwords file. 

2. \-tc: Minimal complexity of the word required to consider replacing it with a simpler word. The value needs to be between 0.0 (all words are considered for simplification) and 1.0 (no words are considered for simplification, texts will not be changed). If there are too many words being simplified, many of them you already consider simple enough, consider increasing the *-tc* value. Conversely, if the tool is making too few simplifications and is not replacing many words you consider complex, consider lowering the *-tc* value. The default value is set to 0.2.

3. \-nc: Number of candidate replacement words to consider. The candidates words are those most similar to the target word in the distributional vector space. This parameters allows you to search for the most suitable replacement among a larger or smaller set of candidate words (by default, 10 candidates are retrieved and considered). 

4. \-st:  The minimal cosine similarity between the embedding of the original word and the embedding of the candidate replacement word that is needed for replacement to occur. The value needs to be between 0.0 (the best candidate replacement will always replace the original word) and 1.0 (the best candidate replacement will never replace the original word, texts will not be changed). If there are too many words being simplified, many of them being replaced with semantically related (but not similar) words, consider increasing the *-st* value. Conversely, if too few simplifications are being made, consider lowering the *-st* value. The deafult value is 0.55, but the optimal value directly depends on the concrete distributional embedding space you use, and needs to be tuned independently for any space of pre-trained word embeddings. 

5. \-cd: The minimal drop in complexity that would be achieved by replacing the source text word with a replacement candidate. Recommended values are between between 0.0 (any complexity reduction is good enough) and 0.1 (the replacement word must be at least 10 percent simpler). If there are too many words being simplified consider increasing the *-cd* value. Conversely, if the tool is making too few simplifications, consider lowering the *-cd* value. The default value is 0.03. 

6. \-w (or \-\-window): The ranking of the candidate simplification words also depends on their similarity with the context surrounding the target word. With this parameter, you determine the size of the context (a symmetric window of size *\-w*, i.e., *\-w* words from each side of the target word are considered for semantic comparison with the candidate simplification). Default value is 5. 

### Prerequisites

- The tool requires the basic libraries from the Python scientific stack: *numpy* (tested with version 1.12.1) and *scipy* (tested with version 0.19.0) 

## Referencing

If you're using the LightLS in your work, please cite the following paper: 

```
@InProceedings{glavavs-vstajner:2015:ACL-IJCNLP,
  author    = {Glava\v{s}, Goran  and  \v{S}tajner, Sanja},
  title     = {Simplifying Lexical Simplification: Do We Need Simplified Corpora?},
  booktitle = {Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)},
  month     = {July},
  year      = {2015},
  address   = {Beijing, China},
  publisher = {Association for Computational Linguistics},
  pages     = {63--68},
  url       = {http://www.aclweb.org/anthology/P15-2011}
}

```








