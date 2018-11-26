from __future__ import division
from math import log,sqrt
from nltk.stem import *
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
from load_map import *
import matplotlib.pyplot as plt
import numpy as np
import operator
import collections
import csv
from scipy import stats

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
	return o_counts[word2wid[word]]


def tw_stemmer(word, stemmed_to_original):
	'''Stems the word using Porter stemmer, unless it is a
	username (starts with @). If so, returns the word unchanged.

	:type word: str
	:param word: the word to be stemmed
	:rtype: str
	:return: the stemmed word
	'''

	if word[0] == '@' or word[0] == '#': # don't stem these
		return word
	else:
		stemmed_word = STEMMER.stem(word)
		stemmed_to_original[stemmed_word].append(word)
		return stemmed_word


def PMI(c_xy, c_x, c_y, N):
	'''Compute the pointwise mutual information using cooccurrence counts.

	:type c_xy: int
	:type c_x: int
	:type c_y: int
	:type N: int
	:param c_xy: coocurrence count of x and y
	:param c_x: occurrence count of x
	:param c_y: occurrence count of y
	:param N: total observation count
	:rtype: float
	:return: the pmi value
	'''

	pmi = log((N * c_xy) / (c_x * c_y), 2)
	return pmi


# Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
	print("Warning: PMI is incorrectly defined")
else:
	print("PMI check passed")


def cos_sim(v0, v1):
	'''Compute the cosine similarity between two sparse vectors.

	:type v0: dict
	:type v1: dict
	:param v0: first sparse vector
	:param v1: second sparse vector
	:rtype: float
	:return: cosine between v0 and v1
	'''

	sum_v0 = sum([v**2 for k, v in v0.items()])
	sum_v1 = sum([v**2 for k, v in v1.items()])

	sum_both = 0
	for k, v in v0.items():
		if k in v1:
			sum_both += v * v1[k]

	cos = sum_both / (np.sqrt(sum_v0) * np.sqrt(sum_v1))
	return cos


def jaccard(v0, v1):
	'''Compute the jaccard similarity between two sparse vectors.

	:type v0: dict
	:type v1: dict
	:param v0: first sparse vector
	:param v1: second sparse vector
	:rtype: float
	:return: jaccard between v0 and v1
	'''

	x = v0.keys()
	y = v1.keys()
	intersection_cardinality = len(set(x) & set(y))
	union_cardinality = len(set(x) | set(y))
	return intersection_cardinality / union_cardinality


def dice_coefficient(v0, v1):
	'''Compute the dice coefficient similarity between two sparse vectors.

	:type v0: dict
	:type v1: dict
	:param v0: first sparse vector
	:param v1: second sparse vector
	:rtype: float
	:return: dice coefficient between v0 and v1
	'''

	x = v0.keys()
	y = v1.keys()
	intersection_cardinality = len(set(x) & set(y))
	return 2 * intersection_cardinality/(len(x) + len(y))


def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
	'''Creates context vectors for the words in wids, using PPMI.
	These should be sparse vectors.

	:type wids: list of int
	:type o_counts: dict
	:type co_counts: dict of dict
	:type tot_count: int
	:param wids: the ids of the words to make vectors for
	:param o_counts: the counts of each word (indexed by id)
	:param co_counts: the cooccurrence counts of each word pair (indexed by ids)
	:param tot_count: the total number of observations
	:rtype: dict
	:return: the context vectors, indexed by word id
	'''

	vectors = {}
	for wid0 in wids:
		vect = {}
		c_x = o_counts[wid0]
		for k, v in co_counts[wid0].items():
			c_y = o_counts[k]
			c_xy = v
			pmi = PMI(c_xy, c_x, c_y, tot_count)
			vect[k] = max(pmi, 0)
		vectors[wid0] = vect
	return vectors


def read_counts(filename, wids):
	'''Reads the counts from file. It returns counts for all words, but to
	save memory it only returns cooccurrence counts for the words
	whose ids are listed in wids.

	:type filename: string
	:type wids: list
	:param filename: where to read info from
	:param wids: a list of word ids
	:returns: occurence counts, cooccurence counts, and tot number of observations
	'''

	o_counts = {} # Occurence counts
	co_counts = {} # Cooccurence counts
	fp = open(filename)
	N = float(next(fp))
	for line in fp:
		line = line.strip().split("\t")
		wid0 = int(line[0])
		o_counts[wid0] = int(line[1])
		if(wid0 in wids):
			co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
	return (o_counts, co_counts, N)


# Matches adjectives to WordNet synset form
adjectives_to_synset = {
	'happy': 'happiness.n.01',
	'delighted': 'delight.n.01',
	'ecstatic': 'ecstasy.n.01',
	'cheerful': 'cheerfulness.n.01',
	'calm': 'calm.n.01',
	'peaceful': 'peace.n.01',
	'relaxed': 'relaxation.n.03',
	'quiet': 'quiet.n.04',
	'serene': 'serenity.n.01',
	'angry': 'anger.n.01',
	'irritated': 'irritation.n.01',
	'enraged': 'rage.n.01',
	'annoyed': 'annoyance.n.01',
	'hateful': 'hate.n.01',
	'sad': 'sadness.n.01',
	'depressed': 'depression.n.01',
	'grieved': 'grief.n.01',
	'unhappy': 'unhappiness.n.01',
	'upset': 'upset.n.01'
}

def create_target_similarities(feelings):
    '''Creates the target similarities dictionaries (arrays) using Path, 
    Leacock-Chodorow and Wu-Palmer Similarities algorithms

	:type feelings: array of strings
	:param feelings: all the feelings word used
	:returns: path_similarities (array and dict), lch_similarities (array and dict),
                wup_similarities (array and dict)
	'''
    
    path_similarities_dict = {}
    lch_similarities_dict = {}
    wup_similarities_dict = {}
    
    path_similarities = []
    lch_similarities = []
    wup_similarities = []
    
    for i, word1 in enumerate(feelings):
    	for j, word2 in enumerate(feelings):
    		if i >= j:
    			continue
    
    		wid1 = word2wid[STEMMER.stem(word1)]
    		wid2 = word2wid[STEMMER.stem(word2)]
    
    		if word1 in adjectives_to_synset:
    			word11 = adjectives_to_synset[word1]
    		else:
    			continue
    
    		if word2 in adjectives_to_synset:
    			word22 = adjectives_to_synset[word2]
    		else:
    			continue
    
    		if word11 != word22:
    			word11 = wn.synset(word11)
    			word22 = wn.synset(word22)
    
    			sim = word11.path_similarity(word22)
    			path_similarities.append((word1, word2, sim))
    			path_similarities_dict[(wid1, wid2)] = sim
    
    			sim = word11.lch_similarity(word22)
    			lch_similarities.append((word1, word2, sim))
    			lch_similarities_dict[(wid1, wid2)] = sim
    
    			sim = word11.wup_similarity(word22)
    			wup_similarities.append((word1, word2, sim))
    			wup_similarities_dict[(wid1, wid2)] = sim
                
    return path_similarities, path_similarities_dict, lch_similarities,\
            lch_similarities_dict, wup_similarities, wup_similarities_dict


def print_sorted_pairs(similarities, o_counts, first=0, last=100):
	'''Sorts the pairs of words by their similarity scores and prints
	out the sorted list from index first to last, along with the
	counts of each word in each pair.

	:type similarities: dict
	:type o_counts: dict
	:type first: int
	:type last: int
	:param similarities: the word id pairs (keys) with similarity scores (values)
	:param o_counts: the counts of each word id
	:param first: index to start printing from
	:param last: index to stop printing
	:return: none
	'''

	if first < 0:
		last = len(similarities)
	for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
		word_pair = (wid2word[pair[0]], wid2word[pair[1]])
		print("{:.3f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
											o_counts[pair[0]],o_counts[pair[1]]))


def print_relation_sentiments_other_words(test_words, word):
    '''Displays the number of cooccurences between other words (like 'osama' 
    and 'obama') and the feelings from the input file

	:type test_words: array of strings
    :type word: sring
	:param test_words: all the words used for testing: feelings + othe words
    :param word: a word from the category 'other words'
	:return: none
	'''
    
    print(word.upper())
    for test_word in test_words:
    	if test_word == word:
    		continue
    
    	try:
    		print(test_word, STEMMER.stem(test_word),co_counts[word2wid[word]][word2wid[STEMMER.stem(test_word)]])
    	except KeyError:
    		print(test_word)


def save_to_csv(similarities, file_name, first=0, last=100):
	'''Sorts the pairs of words by their similarity scores and save the sorted
	list to csv file, along with the counts of each word in each pair.

	:type similarities: dict
	:type file_name: string
	:param similarities: the word id pairs (keys) with similarity scores (values)
	:param last: the name of the csv file which will contain the similarities
	:return: none
	'''

	with open(file_name, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(['Source', 'Target', 'Weight'])
		for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
			writer.writerow([wid2word[pair[0]], wid2word[pair[1]], round(similarities[pair], 3)])


def save_target_similarities_to_csv(similarities, file_name):
	'''Save the target similarities produced using Path, Leacock-Chodorow
	and Wu-Palmer Similarity algorithms to csv file

	:type similarities: dict
	:type file_name: string
	:param similarities: the word id pairs (keys) with similarity scores (values)
	:param last: the name of the csv file which will contain the similarities
	:return: none
	'''
    
	with open(file_name, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(['Word1', 'Word2', 'Similarity'])
		for triple in similarities:
			writer.writerow([triple[0], triple[1], triple[2]])


def freq_v_sim(sims):
    '''Plot and compute (using Spearman's rank correlation coefficient) the 
    relation between similarities and the frequency of words

	:type sims: dict
	:param sims: the word id pairs (keys) with similarity scores (values)
	:return: none
	'''
    
    xs = []
    ys = []
    for pair in sims.items():
        ys.append(pair[1])
        c0 = o_counts[pair[0][0]]
        c1 = o_counts[pair[0][1]]
        xs.append(min(c0,c1))
    plt.clf() # clear previous plots (if any)
    plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
    plt.plot(xs, ys, 'k.') # create the scatter plot
    plt.xlabel('Min Freq')
    plt.ylabel('Similarity')
    print("Freq vs Similarity Spearman correlation = {:.2f}".format(stats.spearmanr(xs,ys)[0]))
    plt.show() #display the set of plots
   
    
def sim_v_sim(sims1, sims2):
    '''Plot and compute (using Spearman's rank correlation coefficient) the 
    relation between two types of similarities

	:type sims1: dict
    :type sims2: dict
	:param sims1: the word id pairs (keys) with similarity scores (values)
    :param sims2: the word id pairs (keys) with similarity scores (values)
	:return: none
	'''
    
    xs = []
    ys = []
    for pair in sims1.items():
        xs.append(pair[1])
        ys.append(sims2[pair[0]])
    plt.clf() # clear previous plots (if any)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, 'k.') # create the scatter plot
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Jaccard index')
    plt.savefig('spearmanr.pdf')
    print("Similarity 1 vs Similarity 2 Spearman correlation = {:.2f}".format(stats.spearmanr(xs,ys)[0]))
    plt.show() #display the set of plots


def make_pairs(items):
	'''Takes a list of items and creates a list of the unique pairs
	with each pair sorted, so that if (a, b) is a pair, (b, a) is not
	also included. Self-pairs (a, a) are also not included.

	:type items: list
	:param items: the list to pair up
	:return: list of pairs

	'''
    
	return [(x, y) for x in items for y in items if x < y]


# Test words for preliminary task
#test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]

test_words = []
spamReader = csv.reader(open('feelings.csv', newline=''), delimiter=',', quotechar='|')
for row in spamReader:
	row_words = [word for word in row]
	test_words.extend(row_words)


stemmed_to_original = collections.defaultdict(list)
stemmed_to_original_large = collections.defaultdict(list)
stemmed_to_original_all = collections.defaultdict(list)

with open("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/wid_word") as fp:
	for line in fp:
		widstr,word=line.rstrip().split("\t")
		wid=int(widstr)
		stemmed_word = STEMMER.stem(word)
		stemmed_to_original_all[stemmed_word].append(word)


stemmed_words = [tw_stemmer(w, stemmed_to_original) for w in test_words]
#stemmed_words = test_words

all_wids = set([word2wid[x] for x in stemmed_words]) # stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs
wid_pairs = make_pairs(all_wids)

#read in the count information
#(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)

#make the word vectors
#vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)




test_words_large = test_words + [STEMMER.stem('osama'), STEMMER.stem('obama')]
stemmed_words_large = [tw_stemmer(w, stemmed_to_original_large) for w in test_words_large]
#stemmed_words = test_words

all_wids_large = set([word2wid[x] for x in stemmed_words_large]) # stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs
wid_pairs_large = make_pairs(all_wids_large)

#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids_large)

#make the word vectors
vectors = create_ppmi_vectors(all_wids_large, o_counts, co_counts, N)






# Print all cooccurences between 'osama' and the feelings ('obama' and feelings)
print('#'*80)
print_relation_sentiments_other_words(test_words, 'osama')
print('#'*80)
print_relation_sentiments_other_words(test_words, 'obama')
print('#'*80)



            
path_similarities, path_similarities_dict, lch_similarities,lch_similarities_dict,\
    wup_similarities, wup_similarities_dict = create_target_similarities(test_words)
    
    
def compute_similarities(vectors, wid_pairs, file_name, sim_func):
    '''Creates the target similarities dictionaries (arrays) using Path, 
    Leacock-Chodorow and Wu-Palmer Similarities algorithms

	:type vectors: array
	:type wid_pairs: list of pairs
	:type file_name: string
	:type sim_func: function
	:param vectors: PPMI vectors
	:param wid_pairs: pairs of wids
	:param file_name: the name of the output file for similarities
	:param sim_func: the function applied to compute the similarities
	:rtype: dict
	:return: the similarities obtained using sim_func
	'''
    
    print("Compute", file_name)
    sims = {(wid0,wid1): sim_func(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
    print("Sort by", file_name)
    save_to_csv(sims, file_name=file_name + '.csv', last=-1)
    
    return sims


c_sims = compute_similarities(vectors, wid_pairs, 'cos_sim', cos_sim)
j_sims = compute_similarities(vectors, wid_pairs, 'jaccard', jaccard)
d_sims = compute_similarities(vectors, wid_pairs, 'dice', dice_coefficient)

c_sims_large = compute_similarities(vectors, wid_pairs_large, 'cos_sim_large', cos_sim)
j_sims_large = compute_similarities(vectors, wid_pairs_large, 'jaccard_large', jaccard)
d_sims_large = compute_similarities(vectors, wid_pairs_large, 'dice_large', dice_coefficient)

save_target_similarities_to_csv(path_similarities, 'path_similarities.csv')
save_target_similarities_to_csv(lch_similarities, 'lch_similarities.csv')
save_target_similarities_to_csv(wup_similarities, 'wup_similarities.csv')

def sort_dict_keys(similarities):
	'''Sorts the elements from the dictionary on the ascending order of the keys

	:type similarities: dict
	:param similarities: a dictionary with similarity scores
	:rtype: dict
	:return: the sorted dictionary
	'''

	result = dict()
	for key_pair, value in similarities.items():
		if key_pair[0] > key_pair[1]:
			new_pair = (key_pair[1], key_pair[0])
		else:
			new_pair = (key_pair[0], key_pair[1])
		result[new_pair] = value
	return result

path_similarities_dict_sorted = sort_dict_keys(path_similarities_dict)
lch_similarities_dict_sorted = sort_dict_keys(lch_similarities_dict)
wup_similarities_dict_sorted = sort_dict_keys(wup_similarities_dict)

def normalize(similarities):
	'''Normalize the values from the dictionary in order to make them fit
	between 0 and 1

	:type similarities: dict
	:param similarities: a dictionary with similarity scores
	:rtype: dict
	:return: a dictionary with normalized values
	'''

	min_val = min(similarities.values())
	max_val = max(similarities.values())
	result = {k: ((v - min_val) / (max_val - min_val)) for k, v in similarities.items()}
	return result

def compute_error(cos_outputs, jaccard_outputs, targets, error_function, to_normalize=False):
	'''Compute error on cosine and jaccard similarities using RMSE

	:type cos_outputs: dict
	:type jaccard_outputs: dict
	:type targets: dict
	:type error_function: function
	:type to_normalize: bool
	:param cos_outputs: a dictionary with cosine similarity scores
	:param jaccard_outputs: a dictionary with jaccard similarity scores
	:param targets: a dictionary with targets similarity scores
	:param error_function: RMSE function
	:param to_normalize: checks if we want to normalize the values or not
	:return: none
	'''

	if to_normalize:
		targets = normalize(targets)

	cos_error = root_mean_squared_error(cos_outputs, targets)
	jaccard_error = root_mean_squared_error(jaccard_outputs, targets)
	print(cos_error, jaccard_error)

def root_mean_squared_error(outputs, targets):
	'''Compute error using root mean squared error (RMSE)

	:type outputs: dict
	:type targets: dict
	:param outputs: a dictionary with output similarity scores
	:param targets: a dictionary with targets similarity scores
	:rtype: float
	:return: the error value
	'''

	suma = 0
	for output_key, target_key in zip(outputs, targets):
		suma += (outputs[output_key] - targets[target_key]) ** 2
	return sqrt(suma / len(targets))

print('#'*80)
path_similarities_errors = compute_error(c_sims, j_sims, path_similarities_dict_sorted, root_mean_squared_error, True)
print('*'*80)
lch_similarities_errors = compute_error(c_sims, j_sims, lch_similarities_dict_sorted, root_mean_squared_error, True)
print('*'*80)
wup_similarities_errors = compute_error(c_sims, j_sims, wup_similarities_dict_sorted, root_mean_squared_error, True)
print('#'*80)

