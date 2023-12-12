#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
import json
import math
import re
import itertools
import random
import itertools
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from datasketch import MinHash,MinHashLSH
from datasketch import MinHashLSHForest,MinHashLSHEnsemble
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter,defaultdict
from sklearn.utils import resample
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import resample
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


# In[2]:


# Get the data from the dataset
file_path = 'TVs-all-merged.json' 
import json
with open(file_path, 'r') as file:
    data = json.load(file)

#Get the titles and modelID       
product_titles = []
model_ids = []          

for products in data.values():
    for product in products:
        product_titles.append(product['title'])
        model_ids.append(product['modelID'])

original_title_to_modelid = dict(zip(product_titles, model_ids))
index_to_modelid = {idx: model_id for idx, model_id in enumerate(model_ids)}


# In[3]:


def normalize_terms(text):
    
    inch_patterns = [r'\s*[\W_]*inch', r'\s*[\W_]*inches', r'\s*[\W_]*”', r'\s*[\W_]*-inch', r'\s*[\W_]* inch']
    hz_patterns = [r'\s*[\W_]*Hertz', r'\s*[\W_]*hertz', r'\s*[\W_]*Hz', r'\s*[\W_]*HZ', r'\s*[\W_]* hz', r'\s*[\W_]*-hz', r'\s*[\W_]*hz']
    
    for pattern in inch_patterns:
        text = re.sub(pattern, '', text)
    for pattern in hz_patterns:
        text = re.sub(pattern, 'hz', text)
        
    return text

#Preprocess the title
def preprocess_title(title):
    title = title.lower()
    title = re.sub(r'[^\w\s]', '', title)
    title = title.replace('hdtv', 'hd')
    title = title.replace('ledlcd', 'led lcd')
    title = title.replace(" ", "")  # Remove all spaces 

    title = normalize_terms(title)

    return title

preprocessed_titles = [preprocess_title(product['title']) for products in data.values() for product in products]
preprocessed_titles_dict = {i: title for i, title in enumerate(preprocessed_titles)}

#Preprocess modelID
def preprocess_modelID(modelID):
    modelID = modelID.lower()
    modelID = re.sub(r'[^\w\s]', '', modelID)
    modelID = modelID.replace('hdtv', 'hd')
    modelID = modelID.replace('ledlcd', 'led lcd')
    #modelID = modelID.replace(" ", "")  # Remove all spaces (added)

    modelID = normalize_terms(modelID)

    return modelID

preprocessed_modelID = [preprocess_modelID(product['modelID']) for products in data.values() for product in products]


# In[4]:


#Generate shingles from the title
def generate_shingles(title):
    title = normalize_terms(title)
    shingles = set()
    for i in range(len(title) - 8):  
        shingle = title[i:i + 9]  
        shingles.add(shingle)
    return shingles

print(generate_shingles(preprocessed_titles_dict[0]))


# In[5]:


#Compute minhash signatures
num_perm = 500

def generate_signature_matrix(preprocessed_titles, num_perm):
    minhashes = [MinHash(num_perm=num_perm) for _ in preprocessed_titles]

    for i, title in enumerate(preprocessed_titles):
        for shingle in generate_shingles(title):
            minhashes[i].update(shingle.encode('utf-8'))

    signature_matrix = np.array([[minhash.hashvalues[i] for minhash in minhashes] for i in range(num_perm)])
    
    return signature_matrix


# In[6]:


def hash_vector_to_bucket(vector, num_buckets=2**24): 

    hash_value = hash(tuple(vector))
    return hash_value % num_buckets

def apply_lsh(signature_matrix, b, r, num_buckets=2**24):

    n, num_titles = signature_matrix.shape
    assert n == b * r

    candidate_pairs = set()
    for band in range(b):
        buckets = defaultdict(list)
        start_row = band * r
        end_row = start_row + r
      
        for c in range(num_titles):
            column_slice = signature_matrix[start_row:end_row, c]
            bucket = hash_vector_to_bucket(column_slice, num_buckets)
            buckets[bucket].append(c)

      
        for bucket_items in buckets.values():
            if len(bucket_items) > 1:
                for pair in itertools.combinations(bucket_items, 2):
                    candidate_pairs.add(pair)

    return candidate_pairs


# In[7]:


# Calculate the actual number of duplicates
modelID_to_indices = {}
for index, model_id in index_to_modelid.items():
    if model_id not in modelID_to_indices:
        modelID_to_indices[model_id] = []
    modelID_to_indices[model_id].append(index)

total_actual_duplicates = 0

for model_id, indices in modelID_to_indices.items():
    if len(indices) > 1:
        total_actual_duplicates += 1

# Number of true duplicates
print(f"Total number of actual duplicates: {total_actual_duplicates}")


# In[8]:


#Jaccard Similarity
def estimate_jaccard_similarity(pair_indices, signature_matrix):
    # Get the signatures for the pair of indices
    signature1 = signature_matrix[:, pair_indices[0]]
    signature2 = signature_matrix[:, pair_indices[1]]

    # Calculate the Jaccard similarity
    intersect = sum(signature1 == signature2)
    union = len(signature1)
    jaccard_similarity = intersect / union if union > 0 else 0

    return jaccard_similarity

#True duplicate check
def is_duplicate(pair_indices, model_ids):
    model_id1 = model_ids[pair_indices[0]]
    model_id2 = model_ids[pair_indices[1]]
    
    # Compare the first two model IDs
    if model_id1 != model_id2:
        return False
    
    # Check for equality among other model IDs in pair_indices
    for i in range(2, len(pair_indices)):
        if model_ids[pair_indices[i]] != model_id1:
            return False
    
    return True



# In[9]:


# Determine all possible values for b and r 
n = num_perm
possible_values_b = []
possible_values_r = []

for b in range(1, n + 1):
    if n % b == 0:
        r = n // b
        t = (1 / b) ** (1 / r)
        possible_values_b.append(b)
        possible_values_r.append(r)
        
print(possible_values_b)
print(possible_values_r)


# In[11]:


#Parameter tuning
b_values = possible_values_b
r_values = possible_values_r

precision_scores =[]
recall_scores =[]
f1_scores =[]
f1_star_scores = []
n_candidates = []
comparisons_total = []
frac_of_comp_total = []
thresholds = []

alpha = 0.3 #used to determine weight for f1* 

for b, r in zip(b_values, r_values):
    t = (1/b)**(1/r)
    thresholds.append(t*100)
    #print("t = ", t)
    
    signature_matrix = generate_signature_matrix(preprocessed_titles, num_perm)
    candidate_pairs = apply_lsh(signature_matrix, b, r)
    print(len(candidate_pairs))
    
    comparisons, true_positives, false_positives = 0, 0, 0
    for pair_indices in candidate_pairs:
        is_dup = is_duplicate(pair_indices, preprocessed_modelID)
        jaccard = estimate_jaccard_similarity(pair_indices, signature_matrix)
        if jaccard > 0.4: 
            comparisons +=1
            
            if is_dup:
                true_positives += 1
            else:
                false_positives += 1
                
    print(comparisons, true_positives, false_positives)
    
    # Calculate precision, recall, and F1 score
    precision = true_positives / comparisons if comparisons > 0 else 0
    recall = true_positives / total_actual_duplicates if total_actual_duplicates > 0 else 0

    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    f1_star = 2 * ((precision * recall) / ((alpha * precision) + ((1 - alpha) * recall))) if ((alpha * precision) + ((1 - alpha) * recall)) > 0 else 0
    
    # Store the scores
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1*100)
    f1_star_scores.append(f1_star*100)
    
    total_titles = (len(preprocessed_titles))
    frac_of_comp = comparisons / ((total_titles * (total_titles - 1))/ 2)
    frac_of_comp_total.append(frac_of_comp)
    
    #print(comparisons, frac_of_comp, precision, recall, f1, f1_star)
    
    print(f"t = {t*100:.3f}, F1: {f1*100:.3f}, F1*: {f1_star*100:.3f}, b:{b}, r:{r}")

#Plotting t vs F1 Score
plt.plot(thresholds, f1_scores, marker='o')
plt.xlabel('Threshold (t)%')
plt.ylabel('F1 Score %')
plt.title('Threshold vs F1 Score for n=500')
plt.grid(True)
plt.show()

#Plotting t vs F1* Score
plt.plot(thresholds, f1_star_scores, marker='o')
plt.xlabel('Threshold (t)%')
plt.ylabel('F1* Score %')
plt.title('Threshold vs F1* Score for n=500')
plt.grid(True)
plt.show()


# In[25]:


#Bootstrapping 
b = possible_values_b
r = possible_values_r

bootstrap = 5
alpha = 0.3

pq_avg =[]
pc_avg =[]
f1_avg =[]
f1_star_avg = []
    
frac_of_comp_avg = []

for b, r in zip(b_values, r_values):
    t = (1/b)**(1/r)
    thresholds.append(t*100)
    
    pq_scores =[]
    pc_scores =[]
    f1_scores =[]
    f1_star_scores = []
    
    comparisons_total = []
    frac_of_comp_total = []

    for i in range(bootstrap):
        
        # Sample 63% of the data randomly with replacement for training
        test_indices = random.sample(range(len(preprocessed_titles)), int(0.6 * len(preprocessed_titles)))
        test_titles = [preprocessed_titles[idx] for idx in test_indices]
    
        # Generate signature matrix for test data
        signature_matrix = generate_signature_matrix(test_titles, num_perm)
            
        # Apply LSH on test data to find candidate pairs
        candidate_pairs = apply_lsh(signature_matrix, b, r)
    
        modelID_test = [preprocessed_modelID[idx] for idx in test_indices]
    
        # Calculate true positives, false positives, and total actual duplicates based on test set
        comparisons, true_positives, false_positives = 0, 0, 0
        for pair_indices in candidate_pairs:
            is_dup = is_duplicate(pair_indices, modelID_test)
            jaccard = estimate_jaccard_similarity(pair_indices, signature_matrix)
            if jaccard > 0.4: 
                comparisons +=1
            
                if is_dup:
                    true_positives += 1
                else:
                    false_positives += 1
                
        #print(comparisons, true_positives, false_positives)
        
        # Calculate total actual duplicates in the test set
        total_actual_duplicates = 0
        for model_id, indices in modelID_to_indices.items():
            if all(idx in test_indices for idx in indices):
                total_actual_duplicates += 1
        #print(total_actual_duplicates)
    
        # Calculate pq, pc, recall, F1 and F1*
        pq = true_positives / comparisons if comparisons > 0 else 0
        pc = true_positives / total_actual_duplicates if total_actual_duplicates > 0 else 0
        f1 = 2 * (pq * pc) / (pq + pc) if (pq + pc) > 0 else 0
        f1_star = 2 * (pq * pc) / ((alpha * pq) + ((1 - alpha) * pc)) if ((alpha * pq) + ((1 - alpha) * pc)) > 0 else 0
    
        # Store the scores
        pq_scores.append(pq)
        pc_scores.append(pc)
        f1_scores.append(f1)
        f1_star_scores.append(f1_star)
        
        total_titles = (len(test_titles))
        frac_of_comp = comparisons / ((total_titles * (total_titles - 1))/ 2)
        frac_of_comp_total.append(frac_of_comp)

    # Calculate average metrics over all bootstraps
    avg_pq = (sum(pq_scores) / bootstrap)*100
    avg_pc = (sum(pc_scores) / bootstrap)*100
    avg_f1 = (sum(f1_scores) / bootstrap)*100
    avg_f1_star = (sum(f1_star_scores) / bootstrap)*100
    avg_frac_of_comp = (sum(frac_of_comp_total) / bootstrap)*100

    print(f"Average values for t = {t*100:.3f}")
    print(f"Average pq: {avg_pq:.3f}")
    print(f"Average pc: {avg_pc:.3f}")
    print(f"Average F1 score: {avg_f1:.3f}")
    print(f"Average F1* score: {avg_f1_star:.3f}")
    print(f"Average Fraction Comparison: {avg_frac_of_comp:.3f}")
    print(f"----------------------------------------------")
    
    # Store averages per treshhold
    pq_avg.append(avg_pq)
    pc_avg.append(avg_pc)
    f1_avg.append(avg_f1)
    f1_star_avg.append(avg_f1_star)
    frac_of_comp_avg.append(avg_frac_of_comp)
    


# In[24]:


plt.plot(frac_of_comp_avg, pq_avg, marker='o')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Quality')
#plt.title('F1 vs Fraction of Comparisons')
plt.grid(True)
plt.show()

plt.plot(frac_of_comp_avg, pc_avg, marker='o')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Completeness')
#plt.title('F1* vs Fraction of Comparisons')
plt.grid(True)
plt.show()

plt.plot(frac_of_comp_avg, f1_avg, marker='o')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1 score')
#plt.title('pq vs Fraction of Comparisons')
plt.grid(True)
plt.show()

plt.plot(frac_of_comp_avg, f1_star_avg, marker='o')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1* score')
#plt.title('pq vs Fraction of Comparisons')
plt.grid(True)
plt.show()

