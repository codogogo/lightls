from simplification import lightls
from helpers import io_helper
from embeddings import text_embeddings
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser(description='A light-weight language-agnostic tool for lexical text simplification.')
parser.add_argument('datadir', help='Path to the directory containing the files with texts to be simplified.')
parser.add_argument('outdir', help='Path to directory in which the lexically simplified texts are to be stored (together with the files containing the lists of substitutions made).')
parser.add_argument('wordfreqs', help='Path to the file containing the precomputed word frequencies in a large corpus (one pair word-frequency per line, word whitespace separated from its frequency).')
parser.add_argument('embs', help='Path to the file containing pre-trained word embeddings')
parser.add_argument('-s', '--stopwords', help='Path to the file containing the list of stopwords for the source language.')
parser.add_argument('-tc', '--tholdcmplx', type=float, help='The minimal complexity of the word needed to consider replacing it with a simpler word. The value needs to be between 0.0 (all words are considered for simplification) and 1.0 (no words are considered for simplification, texts will not be changed), default = 0.2', default = 0.2)
parser.add_argument('-nc', '--numcands', type=int, help='Number of candidate replacement words to consider, for source words with complexity above the -tc value (default 10)', default = 10)
parser.add_argument('-st', '--tholdsim', type=float, help='The minimal cosine similarity between the embeddings of the original word and the candidate replacement word required for replacement. The value needs to be between 0.0 (the best candidate replacement will always replace the original word) and 1.0 (the best candidate replacement will never replace the original word, texts will not be changed), default = 0.55', default=10)
parser.add_argument('-cd', '--dropcmplx', type=float, help='The minimal drop in complexity that would be achieved by replacing the source text word with a replacement candidate. Recommended values are between between 0.0 (any complexity reduction is good enough) and 0.1 (the replacement word must be at least 10 percent simpler), default = 0.025', default=0.025)
parser.add_argument('-w', '--window', type=int, help='The size of the symmetric window around the original word considered for simplification defining the contextual words whose similarity with the replacement candidates is to be measured (contextual similarity features, default = 5)', default=5)
	
args = parser.parse_args()

if not os.path.isdir(os.path.dirname(args.datadir)):
	print("Error: Directory containing the input files not found.")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.outdir)):
	print("Error: Output directory not found.")
	exit(code = 1)

if not os.path.isfile(args.embs):
	print("Error: File containing pre-trained word embeddings not found.")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.wordfreqs)):
	print("Error: File containing word frequencies (pre-computed from a large corpus) not found.")
	exit(code = 1)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Starting lexical simplification.", flush = True)
files = io_helper.load_all_files(args.datadir)
filenames = [x[0] for x in files]
texts = [x[1] for x in files]

t_embeddings = text_embeddings.Embeddings()
t_embeddings.load_embeddings(args.embs, 200000, language = 'default', print_loading = True, skip_first_line = True, normalize = True)
t_embeddings.inverse_vocabularies()

print("Loading unigram frequencies...")
ls = io_helper.load_lines(args.wordfreqs)
wfs = {x.split()[0].strip() : int(x.split()[1].strip()) for x in ls}

parameters = {"complexity_drop_threshold" : args.dropcmplx, "num_cand" : args.numcands, "similarity_threshold" : args.tholdsim, "context_window_size" : args.window, "complexity_threshold" : args.tholdcmplx}
print("Parameters: ")
print(parameters)

stopwords = io_helper.load_lines(args.stopwords) if args.stopwords else None

simplifier = lightls.LightLS(t_embeddings, wfs, parameters, stopwords)

for i in range(len(filenames)):
	print("Simplifying text in file: " + str(filenames[i]) + "(" + str(i+1) + "/" + str(len(filenames)) + ")")
	simp_text, subs = simplifier.simplify_text(texts[i])	
	io_helper.write_list(args.outdir + "/" + os.path.basename(filenames[i]), [simp_text])
	io_helper.write_list_tuples_separated(args.outdir + "/" + os.path.splitext(os.path.basename(filenames[i]))[0] + ".subs", subs)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Lexical simplification completed. I'm out of here, ciao bella!", flush = True)