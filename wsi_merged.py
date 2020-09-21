import os
import sys
sys.path.insert(0, os.path.abspath('/dbfs/FileStore/word_sense_induction'))
sys.path.insert(0, os.path.abspath('/Users/Jimmy/ECDICT'))

import re
from stardict import StarDict
import pickle
from collections import defaultdict
from heapq import *
import opencc
import pkuseg

# PRINT OPTIONS FOR DEBUGGING
PRINT_SENTENCE = False
PRINT_SPACING_BW_SENTENCE = False

PRINT_CANDIDATES = False

PRINT_TARGET_TRANSLATIONS = False
PRINT_CANDIDATES_TRANSLATIONS = False

PRINT_FB_EN = False
PRINT_FB_ZH = False
PRINT_WINNER = False

RUNNING_ON_DATABRICKS = False



def WSI(sentences, target_word, FB_en, FB_zh, EC_dict, converter, nlp, seg):
    '''
    Algorithm for Word Sense Induction:
    1. Get FB map of "cold" from FB_en
    2. For each word in sentence, check FB_en[cold], and return the word `ww` with 
       strongest degree (most frequent word_pair)
    3. For each sense of "cold" and each sense of `ww`:
        * Check FB_zh for their frequency
        * Return the most frequent pair (w1, w2)
        * Now we have Chinese sense labels for both words
    4. Group sentences according to the Chinese labels we obtain
    '''
    def get_translations(word, EC_Dict, converter, nlp, seg):    
        ''' 
        Helper method to get all translations returned by the English-Chinese dictionary.
        word: string = the word to look up
        EC_Dict: Star_Dict  = English-Chinese Dictionary
        '''
        def remove_pos(definition):
            pos_set = set(['aux. ', 'num. ', 'art. ', 'pron. ', 'adv. ', 'a. ', 'interj. ', 'conj. ', 'prep. ', 'v. ', 'vi. ', 'vt. ', 'n. '])
            for pos in pos_set:
                if definition.startswith(pos):
                    return re.sub(pos, "", definition, 1)
            return definition

        def remove_parentheses(definition):
            return re.sub(r"[\(（].+[\)）]", "", definition)
           
        def remove_brackets(definition):
            return re.sub(r"\[.+\]\s?", "", definition)

        def simplified2traditional(word, converter):
            return converter.convert(word)

        def cross_platform_newline(text):
            if "\n" in text:
                return text.replace("\r", "")
            else:
                return text.replace("\r", "\n")

        def standardize_split_tokens(text):
            return re.sub(r"[；:,;]\s?", "<SPLIT>", text)
        
        translations = set()

        lemma = nlp(word)[0].lemma_        
        lemma_query = EC_Dict.query(lemma)
        word_query = EC_Dict.query(word)
        # If the word does not exist in the dictionary, return the empty set
        if not lemma_query and not word_query:
            return translations

        # Skip checking duplicates
        if word == lemma:
            lemma_query = None

        # Join both (word and lemma) queries
        if lemma_query:
            lemma_translation = lemma_query['translation']
            lemma_translation = cross_platform_newline(lemma_translation)
        
        if word_query:
            word_translation = word_query['translation']
            word_translation = cross_platform_newline(word_translation)
        
        if lemma_query and word_query:
            translation = word_translation + '\n' + lemma_translation
        elif lemma_query:
            translation = lemma_translation
        else:
            translation = word_translation      

        # Extract definitions
        print(translation)
        translation = standardize_split_tokens(translation)
        print(translation)
        print(' ')

        definitions = translation.split('\n')
        for definition in definitions:
            for fine_grained_def in definition.split('<SPLIT>'):
                fine_grained_def = remove_brackets(fine_grained_def)
                fine_grained_def = remove_pos(fine_grained_def)
                fine_grained_def = re.sub(r"(.*)[的地]", r"\g<1>", fine_grained_def, 1)

                fine_grained_def = remove_parentheses(fine_grained_def)

                traditional_zh_definition = simplified2traditional(fine_grained_def, converter)            
                if len(traditional_zh_definition) > 0:
                    translations.add((traditional_zh_definition, definition))
     
        return translations
    
    def get_head_and_children(sentence, target_word, nlp):
        doc = nlp(sentence)
        candidates = set()
        for word in doc:
            if word.text == target_word:
                head = word.head
                if not head.is_stop:
                    candidates.add(head.text)
                for child in word.children:
                    if not child.is_stop:
                        candidates.add(child.text)
                break
        return candidates

    def get_topK_frequent_words(words, k, target_word_FB, target_word):
        heap = []
        for word in words:
            if word == target_word:
                continue
            elif word.lower() in ['has', 'have', 'having', 'had', 'being', 'is', 'are', 'am', 'were', 'was']:
                continue
            degree = 0 if word not in target_word_FB else target_word_FB[word]
            heappush(heap, (-degree, word)) # max heap
        
        if PRINT_FB_EN:
            print('FB_en scores: ', sorted(heap))
        result = set()
        prev_degree = heap[0][0]
        while heap and (len(result) < k or heap[0][0] == prev_degree):
            degree, word = heappop(heap)
            if word not in result:
                result.add(word)
            prev_degree = degree
        return result

    
    def get_most_frequent_definition(FB_zh, target_word_translations, candidate_word_translations):
        if PRINT_CANDIDATES_TRANSLATIONS:
            print('candidate_word_translations: ', candidate_word_translations)        
        if PRINT_TARGET_TRANSLATIONS:
            print('target_word_translations: ', target_word_translations)
        heap = []
        for target_word_translation in target_word_translations:
            for candidate_word_translation in candidate_word_translations:
                # Skip over translations not found in FB_zh
                if target_word_translation[0] not in FB_zh:
                    continue
                if candidate_word_translation[0] not in FB_zh[target_word_translation[0]]:
                    continue
                freq = FB_zh[target_word_translation[0]][candidate_word_translation[0]]
                heappush(heap, (-freq, (target_word_translation, candidate_word_translation)))
        
        if PRINT_FB_ZH:
            print('FB_zh scores: ', sorted(heap))
        freq, word_pair = heap[0]
        if PRINT_WINNER:
            print(word_pair[0])

        target_word_translation, candidate_word_translation = word_pair
        return target_word_translation[1]


    # Step 1: Get FB map of target word (Spark SQL)
    if RUNNING_ON_DATABRICKS:
        target_word_FB = FB_en.select("Friends").where(FB_en.Word == target_word).rdd.collect()[0][0]
    else:
        target_word_FB = FB_en

    labels = {}
    debug_labels = {}
    for (sentence, inst_id) in sentences:
        if PRINT_SENTENCE:
            print('Sentence: ', sentence)
        # Step 2: Get the top K frequent word pair
        words = sentence.split()
        k = 3
        head_and_children = get_head_and_children(sentence, target_word, nlp)
        candidates = get_topK_frequent_words(words, k, target_word_FB, target_word)
        if PRINT_CANDIDATES:
            print('candidate_words: ', candidates)
            print('head_and_children: ', head_and_children)
        candidates = candidates.union(head_and_children)
        
        # Step 3: Get all the translations, and return the most frequent definition according to FB_zh[target_word]
        target_word_translations = get_translations(target_word, EC_dict, converter, nlp, seg)
        all_candidate_word_translations = []

        for candidate_word in candidates:
            candidate_word_translations = get_translations(candidate_word, EC_dict, converter, nlp, seg)
            all_candidate_word_translations.extend(candidate_word_translations)

        # Should be `感冒` for "I caught a `cold`"
        try:
            definition = get_most_frequent_definition(FB_zh, target_word_translations, all_candidate_word_translations)
        except:
            try:
                definition = target_word_translations[0][0]
            except:
                definition = "TARGET WORD TRANSLATION IS BROKEN"
            print("Error: Empty heap!")
            print("Sentence: ", sentence)
            print('candidate_words: ', candidate_words)
            print('target_word_translations: ', target_word_translations)
            print('candidate_word_translations: ', candidate_word_translations)
            
        # Step 4: Group sentences according to the Chinese labels we obtain
        debug_labels[sentence] = definition
        labels[inst_id] = definition
        if PRINT_SPACING_BW_SENTENCE:
            print(' ')
            print(' ')
        
    return labels, debug_labels

# Chinese word segmentation
seg = pkuseg.pkuseg()

# Spacy
import spacy
nlp = spacy.load('en')

# Simplified to Traditional Chinese converter
# 簡體到繁體（臺灣正體標準）並轉換爲臺灣常用詞彙 (see: https://github.com/BYVoid/OpenCC)
converter = opencc.OpenCC('s2twp.json')

# Load Dataset
from collections import defaultdict
eval_dataset = defaultdict(list)
if RUNNING_ON_DATABRICKS:
    dataset_path = '/dbfs/FileStore/tables/2010'
else:
    dataset_path = '/Users/Jimmy/WSI/2010_small'
with open(dataset_path, 'r') as f:
    for line in f:
        line = line.strip()
        sentence, target, inst_id = line.split('\t')
        eval_dataset[target].append((sentence, inst_id))
print("Dataset 2010 loaded.")

# Load English-Chinese Dictionary
if RUNNING_ON_DATABRICKS:     
    db = '/dbfs/FileStore/word_sense_induction/ecdict.db'
else:
    db = 'ecdict.db'
EC_dict = StarDict(db, False)
print('EC_dict loaded.')

# Load Chinese FB
if RUNNING_ON_DATABRICKS:
    path = '/dbfs/FileStore/merged_new.pkl'
else:
    path = 'merged_new.pkl'
FB_zh = pickle.load(open(path, 'rb'))
print("FB_zh loaded.")

# Load English FB
if RUNNING_ON_DATABRICKS:
    from pyspark.sql import SQLContext
    sqlContext = SQLContext(sc)
    FB_en = sqlContext.read.parquet('/FileStore/FB_fixed.parquet').persist()
    print("FB_en loaded.")
else:
    FB_en = {}
    with open('flight_friends.txt', 'r') as f:
        for line in f:
            line = line.strip()
            friend, degree = line.split('\t')
            degree = int(degree)
            FB_en[friend] = degree
    print('FB_en for flight (local) loaded.')


fout_key = open('local_2010_fout_key', 'w')
fout_debug = open('local_2010_fout_debug', 'w')
for target_word, sentences in eval_dataset.items():
    print('TARGET WORD: ', target_word)
    labels, debug_labels = WSI(sentences, target_word, FB_en, FB_zh, EC_dict, converter, nlp, seg)
    for (sentence, label), (instance_id, _) in zip(debug_labels.items(), labels.items()):
        lemma_pos = instance_id.rsplit('.', 1)[0] 
        line = ' '.join([lemma_pos, instance_id, label]) + '\n'
        debug_line = ' '.join([label, sentence]) + '\n'
        
        fout_key.write(line)
        fout_debug.write(debug_line)
