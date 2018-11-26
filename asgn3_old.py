from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from load_map import *
import numpy as np
import collections
import csv

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word, stemmed_to_original):
  '''Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.

  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word

  '''
  if word[0] == '@' or word[0] == '#': #don't stem these
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
  pmi = log((N * c_xy)/(c_x * c_y), 2)
  return pmi

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

def cos_sim(v0,v1):
  '''Compute the cosine similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  '''
  # We recommend that you store the sparse vectors as dictionaries
  # with keys giving the indices of the non-zero entries, and values
  # giving the values at those dimensions.

  #You will need to replace with the real function
  #print (v0)
  sum_v0 = sum([v**2 for k, v in v0.items()])
  sum_v1 = sum([v**2 for k, v in v1.items()])
  
  sum_both = 0
  for k, v in v0.items():
      if k in v1:
          sum_both += v*v1[k]
      
  cos = sum_both / (np.sqrt(sum_v0) * np.sqrt(sum_v1))
  #print("Warning: cos_sim is incorrectly defined")
  return cos


def jaccard(v0, v1):
    x = v0.keys()
    y = v1.keys()
    
    intersection_cardinality = len(set(x) & set(y))
    union_cardinality = len(set(x) | set(y))
    return intersection_cardinality/union_cardinality


def dice_coefficient(v0, v1):
    x = v0.keys()
    y = v1.keys()
    
    intersection_cardinality = len(set(x) & set(y))
    return 2 * intersection_cardinality/(len(x) + len(y))

def random_coefficient(v0, v1):
    return np.random.uniform(0, 1)

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
    #print(len(wids))
    #print(wids)
    #print(o_counts)
    #print(co_counts)
    #print(tot_count)
    vectors = {}
    for wid0 in wids:
        ##you will need to change this
        vect = {}
        c_x = o_counts[wid0]
        for k, v in co_counts[wid0].items():
            c_y = o_counts[k]
            c_xy = v
            pmi = PMI(c_xy, c_x, c_y, tot_count)
            vect[k] = max(pmi, 0)
        vectors[wid0] = vect
        #print(vect)
    #print("Warning: create_ppmi_vectors is incorrectly defined")
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

def print_sorted_pairs(similarities, o_counts, file_name=None, first=0, last=100):
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
  file_handle = None
  if file_name:
      file_handle = open(file_name, 'w+')
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    output_str = "{:.3f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                         o_counts[pair[0]],o_counts[pair[1]])
    output_str += '\n'
    if file_handle:
      file_handle.write(output_str)
    else:
      print(output_str)
  file_handle.close()
  
  
def print_to_csv(similarities, file_name, first=0, last=100):
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

  with open(file_name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['Source', 'Target', 'Weight'])   
      for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
        writer.writerow([wid2word[pair[0]], wid2word[pair[1]], round(similarities[pair], 3)])
  


def print_direct_similarities(similarities, file_name):
  with open(file_name, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['Word1', 'Word2', 'Similarity'])   
      for triple in similarities:
          writer.writerow([triple[0], triple[1], triple[2]])
  
  
  
def freq_v_sim(sims):
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
  print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
#  plt.show() #display the set of plots

def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''
  return [(x, y) for x in items for y in items if x < y]

stemmed_to_original = collections.defaultdict(list)
#test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]

test_words = []
feelings = []
spamReader = csv.reader(open('feelings.csv', newline=''), delimiter=',', quotechar='|')
for row in spamReader:
    row_words = [word for word in row]
    if row_words[0] not in ['obama', 'osama']:
        feelings.extend(row_words)
    test_words.extend(row_words)

stemmed_to_original_all = collections.defaultdict(list)

with open("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/wid_word") as fp:
    for line in fp:
        widstr,word=line.rstrip().split("\t")
        wid=int(widstr)
        stemmed_word = STEMMER.stem(word)
        stemmed_to_original_all[stemmed_word].append(word)


stemmed_words = [tw_stemmer(w, stemmed_to_original) for w in test_words]
#stemmed_words = test_words
all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = make_pairs(all_wids)


#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)

#make the word vectors
vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)
# vectors_tfidf = create_tfidf_vectors(all_wids, o_counts, co_counts, N)




## OSAMA
print('#######################################################################')
print('OSAMA')
for test_word in test_words:
    if test_word == 'osama':
        continue
    try:
        print(test_word, STEMMER.stem(test_word),co_counts[word2wid['osama']][word2wid[STEMMER.stem(test_word)]])
    except KeyError:
        print(test_word)


## OBAMA
print('#######################################################################')
print('OBAMA')
for test_word in test_words:
    if test_word == 'obama':
        continue
    try:
        print(test_word, STEMMER.stem(test_word),co_counts[word2wid['obama']][word2wid[STEMMER.stem(test_word)]])
    except KeyError:
        print(test_word)
print('#######################################################################')
      
adjective_to_noun = {
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

noun_to_adjective = {v: k for k, v in adjective_to_noun.items()}

path_similarities_dict = {}
lch_similarities_dict = {}
wup_similarities_dict = {}

path_similarities = []
lch_similarities = []
wup_similarities = []


end = '.n.01'
for i, word1 in enumerate(feelings):
    #word1_end = '.a.01'
    for j, word2 in enumerate(feelings):
        if i >= j:
          continue
        #word2_end = '.a.01'
        #print (word1, word2)
        #if word1 == 'grief':
        #    word1_end = '.n.01'
        #elif word2 == 'grief':
        #    word2_end = '.n.01'
        
        wid1 = word2wid[STEMMER.stem(word1)]
        wid2 = word2wid[STEMMER.stem(word2)]
        
        if word1 in adjective_to_noun:
          word11 = adjective_to_noun[word1]
        else:
          continue
        
        if word2 in adjective_to_noun:
          word22 = adjective_to_noun[word2]
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
            #print(word1, word2, sim)



# compute cosine similarites for all pairs we consider

print("Compute cos_sim")
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#c_sims = {(wid0,wid1): cos_sim(co_counts[wid0],co_counts[wid1]) for (wid0,wid1) in wid_pairs}

print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts, file_name='Assignment3ANLP/experiments/cos_sim_both.txt', last=-1)
#print_to_csv(c_sims, file_name='Assignment3ANLP/experiments/cos_sim_both.csv', last=-1)
print_to_csv(c_sims, file_name='Assignment3ANLP/experiments/cos_sim.csv', last=-1)


print("Compute jaccard")
j_sims = {(wid0,wid1): jaccard(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#c_sims = {(wid0,wid1): cos_sim(co_counts[wid0],co_counts[wid1]) for (wid0,wid1) in wid_pairs}

print("Sort by jaccard similarity")
print_sorted_pairs(j_sims, o_counts, file_name='Assignment3ANLP/experiments/jaccard_both.txt', last=-1)
#print_to_csv(j_sims, file_name='Assignment3ANLP/experiments/jaccard_both.csv', last=-1)
print_to_csv(j_sims, file_name='Assignment3ANLP/experiments/jaccard.csv', last=-1)


print("Compute dice coefficient")
d_sims = {(wid0,wid1): dice_coefficient(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#c_sims = {(wid0,wid1): cos_sim(co_counts[wid0],co_counts[wid1]) for (wid0,wid1) in wid_pairs}

random_sims = {(wid0,wid1): random_coefficient(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

print("Sort by dice-coefficient similarity")
print_sorted_pairs(d_sims, o_counts, file_name='Assignment3ANLP/experiments/dice_coefficient_both.txt', last=-1)

print_direct_similarities(path_similarities, 'path_similarities.csv')
print_direct_similarities(lch_similarities, 'lch_similarities.csv')
print_direct_similarities(wup_similarities, 'wup_similarities.csv')

def sort_dict_keys(this_dict):
    result_dict = dict()  
    for key_pair, value in this_dict.items():
        if key_pair[0] > key_pair[1]:
            new_pair = (key_pair[1], key_pair[0])
        else:
            new_pair = (key_pair[0], key_pair[1])
        result_dict[new_pair] = value
    return result_dict
    #return {(k2, k1): v if k1 > k2 else (k1, k2): v for (k1, k2), v in this_dict.items()}


path_similarities_dict_sorted = sort_dict_keys(path_similarities_dict)
lch_similarities_dict_sorted = sort_dict_keys(lch_similarities_dict)
wup_similarities_dict_sorted = sort_dict_keys(wup_similarities_dict)

def standardize(xx):
    def compute_mean(zz):
        suma = 0
        for elem in list(zz.values()):
            suma += elem
        return suma / len(zz)
  
    def compute_var(zz, zz_mean):
        suma = 0
        for elem in list(zz.values()):
            suma += (elem - zz_mean) ** 2
        return suma / (len(zz) - 1)
  
    xx_mean = compute_mean(xx)
    xx_var = compute_var(xx, xx_mean)
    normalized_xx = {}
    for k, v in xx.items():
        normalized_xx[k] = (v - xx_mean) / xx_var
    return normalized_xx
  
def normalize(xx):
    min_val = min(xx.values())
    max_val = max(xx.values())
    
    norm = {k: ((v - min_val) / (max_val - min_val)) for k, v in xx.items()}
    #norm = {k: v / max_val for k, v in xx.items()}
    
    return norm
  
def compute_error(outputs, jaccard_outputs, targets, error_function, to_normalize=False):
    #n = outputs / 3.63758615972639
    #normalized_outputs = outputs#standardize(outputs)
    #print(normalized_outputs)
    #targets = {k: ((v - 0.747214401830221) / (3.63758615972639 - 0.747214401830221)) for k, v in targets.items()}
    #normalized_targets= targets#standardize(targets)
    #print(targets)
    #print('#'*20)
    #print(normalized_targets)
    print('*'*10)
    #print(normalized_targets)
    
    
    if to_normalize:
        targets = normalize(targets)
    
    error = root_mean_squared_error(outputs, targets)
    error_jaccard = root_mean_squared_error(jaccard_outputs, targets)
    '''
    for target_key, target_value in targets.items():
        print(target_key)
        target_key_reverse = target_key
        if target_key[0] > target_key[1]:
            target_key_reverse = target_key[1], target_key[0]
        if target_key_reverse in outputs:
            print(target_value, outputs[target_key_reverse], jaccard_outputs[target_key_reverse])
    '''

    print(error, error_jaccard)
        

def root_mean_squared_error(outputs, targets):
    suma = 0
    for output_key, target_key in zip(outputs, targets):
        suma += (outputs[output_key] - targets[target_key]) ** 2
    return sqrt(suma / len(targets))

path_similarities_errors = compute_error(c_sims, j_sims, path_similarities_dict_sorted, root_mean_squared_error, True)
lch_similarities_errors = compute_error(c_sims, j_sims, lch_similarities_dict_sorted, root_mean_squared_error, True)
wup_similarities_errors = compute_error(c_sims, j_sims, wup_similarities_dict_sorted, root_mean_squared_error, True)


'''
for k, v in adjective_to_noun.items():
    word = wn.synset(v)
    print (v.upper(), word.definition())
'''


for k in adjective_to_noun.keys():
    try:
        co_counts[word2wid['enrag']][word2wid[STEMMER.stem(word)]]
    except:
        print (k)














'''
print("Compute cos_sim")
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#c_sims = {(wid0,wid1): cos_sim(co_counts[wid0],co_counts[wid1]) for (wid0,wid1) in wid_pairs}

print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts, file_name='not_pmi_cos_sim.txt', last=-1)


print("Compute jaccard")
j_sims = {(wid0,wid1): jaccard(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#c_sims = {(wid0,wid1): cos_sim(co_counts[wid0],co_counts[wid1]) for (wid0,wid1) in wid_pairs}

print("Sort by jaccard similarity")
print_sorted_pairs(j_sims, o_counts, file_name='not_pmi_jaccard.txt', last=-1)


print("Compute dice coefficient")
d_sims = {(wid0,wid1): dice_coefficient(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#c_sims = {(wid0,wid1): cos_sim(co_counts[wid0],co_counts[wid1]) for (wid0,wid1) in wid_pairs}


print("Sort by dice-coefficient similarity")
print_sorted_pairs(d_sims, o_counts, file_name='not_pmi_dice_coefficient.txt', last=-1)
'''
