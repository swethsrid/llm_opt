from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
import argparse
import os

from typing import Union, Dict
from collections.abc import Iterable

import math
import logging
from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = 'tfidf.pickle'
INDEX_PATH = 'index.pickle'
QN_PATH = 'questions.pickle'
ANS_PATH = 'answers.pickle'

import os

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize
from guesser import print_guess, Guesser

from sklearn.feature_extraction.text import TfidfVectorizer

kTFIDF_TEST_QUESTIONS = {"This capital of England": ['Maine', 'Boston'],
                        "The author of Pride and Prejudice": ['Jane_Austen', 'Jane_Austen'],
                        "The composer of the Magic Flute": ['Wolfgang_Amadeus_Mozart', 'Wolfgang_Amadeus_Mozart'],
                        "The economic law that says 'good money drives out bad'": ["Gresham's_law", "Gresham's_law"],
                        "located outside Boston, the oldest University in the United States": ['College_of_William_&_Mary', 'Rhode_Island']}


class DummyVectorizer:
    """
    A dumb vectorizer that only creates a random matrix instead of something real.
    """
    def __init__(self, width:int=50):
        self.width = width
        self.vocabulary_ = {}
    
    def transform(self, questions: Iterable):
        import numpy as np
        return np.random.rand(len(questions), self.width)

class TfidfGuesser(Guesser):
    """
    Class that, given a query, finds the most similar question to it.
    """
    #def __init__(self, filename:str, min_df:int=10, max_df:float=0.4):
    def __init__(self, filename: str, min_df: int = 4, max_df: float = 0.87,
                 ngram_range: tuple = (1, 2), max_features: int = 50000,
                 use_idf: bool = True, sublinear_tf: bool = True,
                 norm: str = 'l2', use_custom_tokenizer: bool = False):
        """
           filename: filename
           min_df: minimum document frequency
           max_df: maximum document frequency
           ngram_range: tuple of (min_n, max_n) for n-grams 
           max_features: limit vocabulary size for efficiency
           use_idf: inverse document frequency weighting
           sublinear_tf: log scaling for term frequency
           norm: normalization method ('l1', 'l2', or None)
           use_custom_tokenizer: enhanced tokenizer with better preprocessing
        """
        """
        Initializes data structures that will be useful later.

        Args:
           filename: base of filename we store vectorizer and documents to
           min_df: we use the sklearn vectorizer parameters, this for min doc freq
           max_df: we use the sklearn vectorizer parameters, this for max doc freq
        """

        # You'll need add the vectorizer here and replace this fake vectorizer
        #self.tfidf_vectorizer = DummyVectorizer()
        #self.tfidf = None 
        #self.questions = None
        #self.answers = None
        #self.filename = filename
        def custom_tokenizer(text):
            """
            - contractions
            - important punctuation patterns
            - numbers and special terms
            """
    
            import re
            # Preserve year patterns
            text = re.sub(r'\b(\d{4})\b', r'year_\1', text)
            # Preserve important entities (capitalized words often = proper nouns)
            # Convert to lowercase but track if originally capitalized
            tokens = text.split()
            processed = []
            for token in tokens:
                # Remove pure punctuation
                token = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', token)
                if token:
                    processed.append(token.lower())
            return processed
        
        # Build vectorizer with optimized parameters
        vectorizer_params = {
            'min_df': min_df,
            'max_df': max_df,
            'lowercase': True,
            'stop_words': 'english',
            'ngram_range': ngram_range,
            'max_features': max_features,
            'use_idf': use_idf,
            'sublinear_tf': sublinear_tf,
            'norm': norm,
            'strip_accents': 'unicode',
            'analyzer': 'word',
        }
        
        if use_custom_tokenizer:
            vectorizer_params['tokenizer'] = custom_tokenizer
        
        self.tfidf_vectorizer = TfidfVectorizer(**vectorizer_params)
        self.tfidf = None 
        self.questions = None
        self.answers = None
        self.filename = filename
        
        # Store configuration for later reference
        self.config = {
            'min_df': min_df,
            'max_df': max_df,
            'ngram_range': ngram_range,
            'max_features': max_features,
            'use_idf': use_idf,
            'sublinear_tf': sublinear_tf,
            'norm': norm,
        }

    def train(self, training_data, answer_field='page', split_by_sentence=True,
                    min_length=-1, max_length=-1, remove_missing_pages=True):
        """
        The base class (Guesser) populates the questions member, so
        all that's left for this function to do is to create new members
        that have a vectorizer (mapping documents to tf-idf vectors) and
        the matrix representation of the documents (tfidf) consistent
        with that vectorizer.
        """
        
        Guesser.train(self, training_data, answer_field, split_by_sentence, min_length,
                      max_length, remove_missing_pages)

        self.tfidf = self.tfidf_vectorizer.transform(self.questions)
        
        #logging.info("Creating tf-idf dataframe with %i" % len(self.questions))
                      
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        logging.info("Created TF-IDF matrix with %i documents" % len(self.questions))
        logging.info("Vocabulary size: %i features" % vocab_size)
        logging.info("Matrix shape: %s" % str(self.tfidf.shape))
        logging.info("Matrix sparsity: %.2f%%" % (100.0 * (1 - self.tfidf.nnz / (self.tfidf.shape[0] * self.tfidf.shape[1]))))
        
    def save(self):
        """
        Save the parameters to disk
        """
        Guesser.save_questions_and_answers(self)
        
        path = self.filename
        with open("%s.vectorizer.pkl" % path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open("%s.tfidf.pkl" % path, 'wb') as f:
            pickle.dump(self.tfidf, f)

    def __call__(self, question, max_n_guesses):
        """
        Given the text of questions, generate guesses (a list of both both the page id and score) for each one.

        Keyword arguments:
        question -- Raw text of the question
        max_n_guesses -- How many top guesses to return
        """
        top_questions = []
        top_answers = []
        top_sim = []

        # Compute the cosine similarity
        question_tfidf = self.tfidf_vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_tfidf, self.tfidf)
        cos = cosine_similarities[0]

        query_words = set(question.lower().split())
        for i, train_q in enumerate(self.questions):
            # Only boost documents that already have some similarity
            if cos[i] > 0.05:  # Threshold to avoid boosting irrelevant docs
                train_words = set(train_q.lower().split())
                overlap = len(query_words & train_words)
                # Boost by raw overlap count, not ratio (favors longer matches)
                if overlap >= 2:  # At least 2 words must match
                    cos[i] *= (1 + overlap * 0.05)  # Gentler boost: 10% per word
        
        # Sort indices by similarity (descending)    
        indices = cos.argsort()[::-1]

      
        guesses = []
        num_guesses = min(max_n_guesses, len(indices))
        for i in range(max_n_guesses):
            # The line below is wrong but lets the code run for the homework.
            # Remove it or fix it!
            #idx = i
            #guess =  {"question": self.questions[idx], "guess": self.answers[idx],
            #          "confidence": cos[idx]}
            #guesses.append(guess)
            idx = indices[i]
            guess = {
                "question": self.questions[idx], 
                "guess": self.answers[idx],
                "confidence": float(cos[idx])  # Ensure it's a Python float
            }
            guesses.append(guess)
          
        assert len(guesses) <= max_n_guesses, "Too many guesses: %i > %i" % (len(guesses), max_n_guesses)
        return guesses

    def batch_guess(self, questions:Iterable[str], max_n_guesses:int, block_size:int=1024) -> Iterable[Dict[str, Union[str, float]]]:
        """
        The batch_guess function allows you to find the search
        results for multiple questions at once.  This is more efficient
        than running the retriever for each question, finding the
        largest elements, and returning them individually.  

        To understand why, remember that the similarity operation for an
        individual query and the corpus is a dot product, but if we do
        this as a big matrix, we can fit all of the documents at once
        and then compute the matrix as a parallelizable matrix
        multiplication.

        The most complicated part is sorting the resulting similarities,
        which is a good use of the argpartition function from numpy.

        Args:
           questions: the questions we'll produce answers for
           max_n_guesses: number of guesses to return
           block_size: split large lists of questions into arrays of this many rows
        Returns:
        """

        # IMPORTANT NOTE FOR HOMEWORK: you do not need to complete
        # batch_guess.  If you're having trouble with this, just
        # delete the function, and the parent class will emulate the
        # functionality one row at a time.
        
        from math import floor
    
        all_guesses = []

        logging.info("Querying matrix of size %i with block size %i" %
                     (len(questions), block_size))

        # The next line of code is bogus, this needs to be fixed
        # to give you a real answer.
        for start in tqdm(range(0, len(questions), block_size)):
            #stop = start+block_size
            stop = min(start + block_size, len(questions))
  
            block = questions[start:stop]
            logging.info("Block %i to %i (%i elements)" % (start, stop, len(block)))
            block_tfidf = self.tfidf_vectorizer.transform(block)

            # Compute cosine similarities for all questions in the block at once
            cosine_similarities = cosine_similarity(block_tfidf, self.tfidf)

            for question in range(len(block)):
                cos = cosine_similarities[question_idx]
                
                # Efficient top-k retrieval using argpartition
                if max_n_guesses < len(cos):
                    # Get indices of top max_n_guesses elements
                    top_indices = np.argpartition(cos, -max_n_guesses)[-max_n_guesses:]
                    # Sort these top indices by their scores (descending)
                    top_indices = top_indices[np.argsort(cos[top_indices])[::-1]]
                else:
                    # If requesting more guesses than documents, just sort all
                    top_indices = np.argsort(cos)[::-1][:max_n_guesses]
                  
                guesses = []
                #for idx in list(top_hits[question]):
                #    score = 0.0
                #    guesses.append({"guess": self.answers[idx], "confidence": score, "question": self.questions[idx]})
                for idx in top_indices:
                    guesses.append({
                        "guess": self.answers[idx], 
                        "confidence": float(cos[idx]), 
                        "question": self.questions[idx]
                    })
                  
                all_guesses.append(guesses)

        assert len(all_guesses) == len(questions), "Guesses (%i) != questions (%i)" % (len(all_guesses), len(questions))
        return all_guesses
    
    def load(self):
        """
        Load the tf-idf guesser from a file
        """
        
        path = self.filename
        with open("%s.vectorizer.pkl" % path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        with open("%s.tfidf.pkl" % path, 'rb') as f:
            self.tfidf = pickle.load(f)

        self.load_questions_and_answers()

if __name__ == "__main__":
    # Load a tf-idf guesser and run it on some questions
    from params import *
    logging.basicConfig(level=logging.DEBUG)
    
    parser = argparse.ArgumentParser()
    add_general_params(parser)
    add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    
    guesser = load_guesser(flags, load=True)

    questions = list(kTFIDF_TEST_QUESTIONS.keys())
    guesses = guesser.batch_guess(questions, 3, 1024)

    for qq, gg in zip(questions, guesses):
        print("----------------------")
        print(qq, gg)
        #extra things printed
        for g in gg:
          print(f"  {g['guess']}: {g['confidence']:.4f}")
