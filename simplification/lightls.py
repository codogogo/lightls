import math
import numpy as np
from helpers import string_helper

class LightLS(object):
	"""description of class"""
	def __init__(self, embeddings, word_freqs, parameters, stopwords = None, lang = "default"):
		self.stopwords = stopwords
		self.params = parameters
		self.embeddings = embeddings
		self.lang = lang
		
		self.complexities = {x : 1.0 / math.log2(word_freqs[x] + 2) for x in word_freqs }
		max_freq = max(word_freqs.values())
		min_freq = min(word_freqs.values())
		min_complexity = 1.0 / math.log2(max_freq + 2)
		max_complexity = 1.0 / math.log2(min_freq + 2)
		self.complexities = {x : (self.complexities[x] - min_complexity) / (max_complexity - min_complexity) for x in self.complexities}		

	def fix_token(self, token):
		punctuation = [".", ",", "!", ":", "?", ";", "-", ")", "(", "[", "]", "{", "}", "...", "/", "\\", "''", "\"", "'"]
		if token[0] in punctuation and token[-1] in punctuation:	
			return token[1:-1]
		elif token[0] in punctuation:
			return token[1:]
		elif token[-1] in punctuation:
			return token[:-1]
		else:
			return token

	def fix_token_inverse(self, token, simp_token):
		punctuation = [".", ",", "!", ":", "?", ";", "-", ")", "(", "[", "]", "{", "}", "...", "/", "\\", "''", "\"", "'"]
		if token[0] in punctuation and token[-1] in punctuation:	
			return token[0] + simp_token + token[-1]
		elif token[0] in punctuation:
			return token[0] + simp_token
		elif token[-1] in punctuation:
			return simp_token + token[-1]
		else:
			return simp_token
			
	def simplify_text(self, text):
		simplifications = []
		tokens = text.split()
		for i in range(len(tokens)):
			res = self.try_simplify_token(tokens, i)
			if res:
				simplifications.append((i, res))
		
		tokens_simple = []
		tokens_simple.extend(tokens)
		replacements = []
		for s in simplifications:
			tokens_simple[s[0]] = self.fix_token_inverse(tokens[s[0]], s[1])
			replacements.append((s[0], self.fix_token(tokens[s[0]]), s[1]))
		simplified_text = ' '.join(tokens_simple)
		return (simplified_text, replacements)
			
	def try_simplify_token(self, tokens, index):
		target = self.fix_token(tokens[index])
		# Not simplifying proper names
		if str.isupper(target) or str.istitle(target) or str.isnumeric(target):
			return None
		complexity_target = self.complexities[target] if target in self.complexities else (self.complexities[target.lower()] if target.lower() in self.complexities else 1.0)
		# If the word is simple enough, no need to consider replacing it
		if complexity_target <= self.params["complexity_threshold"]:
			return None

		# we're not "simplifying" stopwords
		if self.stopwords and target.lower() in self.stopwords:
			return None

		tvec = self.embeddings.get_vector(self.lang, target)
		if tvec is None:
			tvec = self.embeddings.get_vector(self.lang, target.lower())
		if tvec is not None:
			candidates = self.embeddings.most_similar_fast_cosine(tvec, self.lang, num = self.params["num_cand"], without_first = True)
			simpler_candidates = {}
			for c in candidates:
				# we discard candidates that are derivational morphological variations of the target word
				if c in target or target in c:
					continue
				lcses = string_helper.longest_common_subsequence(c, target)
				if len(lcses) > 0:
					lcs = lcses.pop()
					if len(target) >= 6 and len(c) >= 6 and len(lcs) >= (min(len(c), len(target)) - 3):
						continue
				# don't allow the target word to be replaced by a stopword
				if self.stopwords is not None and c.lower() in self.stopwords:
					continue

				complexity_cand = self.complexities[c] if c in self.complexities else 1.0
				if (complexity_cand < complexity_target) and ((complexity_target - complexity_cand) >= self.params["complexity_drop_threshold"]):
					simpler_candidates[c] = { "complexity_drop" : complexity_target - complexity_cand }
			if len(simpler_candidates) == 0:
				return None

			context_vecs = self.get_context_vectors(tokens, index)
			self.compute_features(tokens, index, tvec, simpler_candidates, context_vecs)
			feats = ["sim", "complexity_drop"] if len(context_vecs) == 0 else ["sim", "complexity_drop", "context"]
			ranks = {}
			for c in simpler_candidates:
				ranks[c] = []
			for f in feats:
				feat_sorted = sorted({c : simpler_candidates[c][f] for c in simpler_candidates}.items(), key=lambda x:x[1])
				for fs in feat_sorted:
					ranks[fs[0]].append(feat_sorted.index(fs))

			ranked_candidates = sorted({c : sum(ranks[c]) for c in ranks}.items(), key=lambda x:x[1])
			best_candidate = ranked_candidates[-1][0]
			if simpler_candidates[best_candidate]["sim"] >= self.params["similarity_threshold"]:
				return best_candidate
			else: 
				return None
		else:
			return None
	
	def compute_features(self, tokens, index, tvec, candidates, context_vecs):
		for c in candidates:
			candidates[c]["sim"] = np.dot(self.embeddings.get_vector(self.lang, c), tvec) 
			if len(context_vecs) > 0:
				csim = 0.0
				for cv in context_vecs:
					csim += np.dot(self.embeddings.get_vector(self.lang, c), cv)
				candidates[c]["context"] = csim

	def get_context_vectors(self, tokens, index):
		window_size = self.params["context_window_size"]
		start = index - window_size if (index - window_size >= 0) else 0
		end = index + window_size if (index + window_size <= len(tokens)) else len(tokens)
		cvecs = []
		for i in range(start, end):
			if i != index:
				cword = self.fix_token(tokens[i])
				cvec = self.embeddings.get_vector(self.lang, cword)
				if cvec is None:
					cvec = self.embeddings.get_vector(self.lang, cword.lower())
				if cvec is not None:
					cvecs.append(cvec)
		return cvecs
	