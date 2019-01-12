import re
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.linalg import svd
from nltk import word_tokenize
import sys
from functools import reduce


class word2prob(object):
	"""Map words to their probabilities."""
	def __init__(self, count_fn):
		"""Initialize a word2prob object.

		Args:
			count_fn: word count file name (one word per line) 
		"""
		self.prob = {}
		total = 0.0

		for line in open(count_fn):
			k,v = line.split()
			v = int(v)
			k = k.lower()

			self.prob[k] = v
			total += v

		self.prob = { k : (self.prob[k] / total) for k in self.prob }
		self.min_prob = min(self.prob.values())
		self.count = total

	def __getitem__(self, w):
		return self.prob.get(w.lower(), self.min_prob)

	def __contains__(self, w):
		return w.lower() in self.prob

	def __len__(self):
		return len(self.prob)

	def vocab(self):
		return iter(self.prob.keys())


class word2vec(object):
	"""Map words to their embeddings."""
	def __init__(self, vector_fn):
		"""Initialize a word2vec object.

		Args:
			vector_fn: embedding file name (one word per line)
		"""
		self.vectors = {}
    
		for line in open(vector_fn):
			line = line.split()

			# skip first line if needed
			if len(line) == 2:
				continue

			word = line[0]
			embedding = np.array([float(val) for val in line[1:]])
			self.vectors[word] = embedding

	def __getitem__(self, w):
		return self.vectors[w]

	def __contains__(self, w):
		return w in self.vectors



class uSIF(object):
	"""Embed sentences using unsupervised smoothed inverse frequency."""
	def __init__(self, vec, prob, n=11, m=5):
		"""Initialize a sent2vec object.

		Variable names (e.g., alpha, a) all carry over from the paper.

		Args:
			vec: word2vec object
			prob: word2prob object
			n: expected random walk length. This is the avg sentence length, which
				should be estimated from a large representative sample. For STS
				tasks, n ~ 11. n should be a positive integer.
			m: number of common discourse vectors (in practice, no more than 5 needed)
		"""
		self.vec = vec
		self.m = m

		if not (isinstance(n, int) and n > 0):
			raise TypeError("n should be a positive integer")
	
		vocab_size = float(len(prob))
		threshold = 1 - (1 - 1/vocab_size) ** n
		alpha = len([ w for w in prob.vocab() if prob[w] > threshold ]) / vocab_size
		Z = 0.5 * vocab_size
		self.a = (1 - alpha)/(alpha * Z)

		self.weight = lambda word: (self.a / (0.5 * self.a + prob[word])) 

	def _to_vec(self, sentence):
		"""Vectorize a given sentence.
		
		Args:
			sentence: a sentence (string) 
		"""
		# regex for non-punctuation
		not_punc = re.compile('.*[A-Za-z0-9].*')

		# preprocess a given token
		def preprocess(t):
			t = t.lower().strip("';.:()").strip('"')
			t = 'not' if t == "n't" else t
			return re.split(r'[-]', t)

		tokens = []

		for token in word_tokenize(sentence):
			if not_punc.match(token):
				tokens = tokens + preprocess(token)
		
		tokens = list(filter(lambda t: t in self.vec, tokens))

		# if no parseable tokens, return a vector of a's        
		if tokens == []:
			return np.zeros(300) + self.a
		else:
			v_t = np.array([ self.vec[t] for t in tokens ])
			v_t = v_t * (1.0 / np.linalg.norm(v_t, axis=0))
			v_t = np.array([ self.weight(t) * v_t[i,:] for i,t in enumerate(tokens) ])
			return np.mean(v_t, axis=0) 

	def embed(self, sentences):
		"""Embed a list of sentences.

		Args:
			sentences: a list of sentences (strings)
		"""
		vectors = [ self._to_vec(s) for s in sentences ]

		if self.m == 0:
			return vectors

		proj = lambda a, b: a.dot(b.transpose()) * b
		svd = TruncatedSVD(n_components=self.m, random_state=0).fit(vectors)	
	
		# remove the weighted projections on the common discourse vectors
		for i in range(self.m):
			lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
			pc = svd.components_[i]
			vectors = [ v_s - lambda_i * proj(v_s, pc) for v_s in vectors ]

		return vectors

	
def test_STS(model):
	"""Test the performance on the STS tasks and print out the results.

	Expected results:
		STS2012: 0.683
		STS2013: 0.661
		STS2014: 0.784
		STS2015: 0.790
		SICK2014: 0.735
		STSBenchmark: 0.795

	Args:
		model: a uSIF object
	""" 
	test_dirs = [
		'STS/STS-data/STS2012-gold/',
		'STS/STS-data/STS2013-gold/',
		'STS/STS-data/STS2014-gold/',
		'STS/STS-data/STS2015-gold/',
		'STS/SICK-data/',
		'STSBenchmark/'
	]

	for td in test_dirs:
		test_fns = filter(lambda fn: '.input.' in fn and fn.endswith('txt'), os.listdir(td))
		scores = []
	
		for fn in test_fns:
			sentences = re.split(r'\t|\n', open(td + fn).read().strip())
			vectors = model.embed(sentences)
			y_hat = [ 1 - cosine(vectors[i], vectors[i+1]) for i in range(0, len(vectors), 2) ]
			y = list(map(float, open(td + fn.replace('input', 'gs')).read().strip().split('\n')))

			score = pearsonr(y, y_hat)[0]
			scores.append(score)

			print(fn, "\t", score)
		
		print(td, np.mean(scores), "\n")


def test_STS_benchmark(model):
	"""
	Test the performance on the STS benchmark (train and test) and print out the results.

	Expected results:
		STSBenchmark/sts-dev.csv 	 0.842
		STSBenchmark/sts-test.csv 	 0.795

	Args:
		model: a uSIF object
	"""
	test_fns = [ 'STSBenchmark/sts-dev.csv', 'STSBenchmark/sts-test.csv' ]

	for fn in test_fns:
		y, y_hat = [], []
		sentences = []

		for line in open(fn):
			similarity, s1, s2 = line.strip().split('\t')[-3:]
			sentences.append(s1)
			sentences.append(s2)
			y.append(float(similarity))
		
		vectors = model.embed(sentences)
		y_hat = [ 1 - cosine(vectors[i], vectors[i+1]) for i in range(0, len(vectors), 2) ]

		score = pearsonr(y, y_hat)[0]
		print(fn, "\t", score)


def get_paranmt_usif():
	"""Return a uSIF embedding model that used pre-trained ParaNMT word vectors."""
	prob = word2prob('enwiki_vocab_min200.txt')
	vec = word2vec('vectors/czeng.txt')
	return uSIF(vec, prob)

