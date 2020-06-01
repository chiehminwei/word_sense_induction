import re
import os
from stardict import StarDict, LemmaDB
import pickle
from collections import defaultdict
from heapq import *
import opencc


def WSI(sentences, target_word, FB_en, FB_zh, EC_dict, converter, lemma):
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
    def get_translations(word, EC_Dict, converter, lemmaDB):    
        ''' 
        Helper method to get all translations returned by the English-Chinese dictionary.
        word: string = the word to look up
        EC_Dict: Star_Dict  = English-Chinese Dictionary
        '''
        def remove_pos(definition):
            pos_set = set(['art. ', 'pron. ', 'adv. ', 'a. ', 'interj. ', 'conj. ', 'prep. ', 'v. ', 'vi. ', 'vt. ', 'n. '])
            for pos in pos_set:
                if definition.startswith(pos):
                    return re.sub(pos, "", definition, 1)
            return definition

        def remove_parentheses(definition):
            return re.sub(r"\[.+\] ", "", definition, 1)

        def simplified2traditional(word, converter):
            return converter.convert(word)
        
        translations = []

        query = EC_Dict.query(word)
        # If the word does not exist in the dictionary, return the empty list
        if not query:
            return translations

        # Query the lemma not the inflected form 
        # (details under '词形变化' at: https://github.com/skywind3000/ECDICT)
        
        # Never mind
        # lemma = lemmaDB.word_stem(word)
        
        lemma = word
        inflections = query['exchange'].split('/')
        if len(inflections[0]) > 0:
            for inflection in inflections:
                inflection_type, inflection_form = inflection.split(':')
                if inflection_type == '0':
                    lemma = inflection_form 

        query = EC_Dict.query(lemma)
        translation = query['translation']

        # Cross-platform newline
        if "\n" in translation:
            translation = translation.replace("\r", "")
        else:
            translation = translation.replace("\r", "\n")

        definitions = translation.split('\n')
        for definition in definitions:
            definition = remove_parentheses(definition)        
            definition = remove_pos(definition)
            if ", " in definition:
                fine_grained_definitions = definition.split(', ')
            else:
                fine_grained_definitions = definition.split('; ')
            for fine_grained_definition in fine_grained_definitions:
                fine_grained_definition = simplified2traditional(fine_grained_definition, converter)
                # Remove "的" or "地"
                fine_grained_definition = re.sub(r"(.+)[的地]", r"\g<1>", fine_grained_definition, 1)
                translations.append(fine_grained_definition)                
        
        return translations
    
    def get_topK_frequent_words(words, k, target_word_FB):
        heap = []
        for word in words:                
            degree = 0 if word not in target_word_FB else target_word_FB[word]
            heappush(heap, (-degree, word)) # max heap
        
        result = []
        prev_degree = None
        for (degree, word) in heap:
            if k > 0 or degree == prev_degree:
                k -= 1
                result.append(word)
            prev_degree = degree
        return result

    

    def get_most_frequent_definition(FB_zh, target_word_translations, candidate_word_translations):
        print('target_word_translations: ', target_word_translations)
        print('candidate_word_translations: ', candidate_word_translations)
        heap = []
        for target_word_translation in target_word_translations:
            for candidate_word_translation in candidate_word_translations:
                # Skip over translations not found in FB_zh
                if target_word_translation not in FB_zh:
                    continue
                if candidate_word_translation not in FB_zh[target_word_translation]:
                    continue
                freq = FB_zh[target_word_translation][candidate_word_translation]
                heappush(heap, (-freq, (target_word_translation, candidate_word_translation)))
        
        print('heap: ', heap)
        freq, word_pair = heap[0]
        target_word_translation, candidate_word_translation = word_pair
        return target_word_translation


    # Step 1: Get FB map of target word (Spark SQL)
    target_word_FB = target_word_FB = FB_en.select("Friends").where(FB_en.Word == target_word).rdd.collect()[0][0]

    labels = {}
    for sentence in sentences:
        # Step 2: Get the top K frequent word pair
        words = sentence.split()
        k = 3
        candidates = get_topK_frequent_words(words, k, target_word_FB)
        print('candidate_words: ', candidates)
        
        # Step 3: Get all the translations, and return the most frequent definition according to FB_zh[target_word]
        target_word_translations = get_translations(target_word, EC_dict, converter, lemma)
        all_candidate_word_translations = []

        for candidate_word in candidates:
            candidate_word_translations = get_translations(candidate_word, EC_dict, converter, lemma)
            all_candidate_word_translations.extend(candidate_word_translations)

        # Should be `感冒` for "I caught a `cold`"
        definition = get_most_frequent_definition(FB_zh, target_word_translations, all_candidate_word_translations)
                    
        # Step 4: Group sentences according to the Chinese labels we obtain
        labels[sentence] = definition

    return labels


db = os.path.join(os.path.dirname(__file__), 'ecdict.db')
EC_dict = StarDict(db, False)
print('EC_dict loaded.')

lemma = LemmaDB()
lemma.load('lemma.en.txt')
print('LemmaDB loaded.')

# 簡體到繁體（臺灣正體標準）並轉換爲臺灣常用詞彙 (see: https://github.com/BYVoid/OpenCC)
converter = opencc.OpenCC('s2twp.json')

path = '/FileStore/merged_new.pkl'
FB_zh = pickle.load(open(path, 'rb'))
print("FB_zh loaded.")

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
FB_en = sqlContext.read.parquet('/FileStore/FB_fixed.parquet')
print("FB_en loaded.")

# Sometimes it works...
target_word = "cold"
sentences = ["I caught a cold", "The weather is cold"]
labels = WSI(sentences, target_word, FB_en, FB_zh, EC_dict, converter, lemma)
print(labels)

# But sometimes it doesn't
target_word = "mouse"
sentences = ["The computer mouse is hard to use", "Mouse and cat"]
labels = WSI(sentences, target_word, FB_en, FB_zh, EC_dict)
print(labels)
