import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import spacy
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
 
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

def get_word_synsets(word):
    synsets = wn.synsets(word)
    return synsets

def calculate_synset_similarity(synset1, synset2):
    similarity = synset1.path_similarity(synset2)
    return similarity

def align_sentences(sent1, sent2,n, threshold=0.3):
    alignments = []

    for i in range(1,n+1):
        if i == 1:
            tokens1 = join_n_grams(sent1,i)
            tokens2 = join_n_grams(sent2,i)
        else:
            tokens1.extend(join_n_grams(sent1,i))
            tokens2.extend(join_n_grams(sent2,i))

    for token1 in tokens1:
        if token1.lower() in nltk.corpus.stopwords.words('english'):
            continue
        
        if token1 in ['.', ',', '?', '!', ':', ';']:
            continue

        synsets1 = get_word_synsets(token1)

        max_similarity = 0

        for token2 in tokens2:
            if token2.lower() in nltk.corpus.stopwords.words('english'):
                continue

            if token2 in ['.', ',', '?', '!', ':', ';']:
                continue

            synsets2 = get_word_synsets(token2)

            for synset1 in synsets1:
                for synset2 in synsets2:
                    similarity = calculate_synset_similarity(synset1, synset2)
                    if similarity is not None and similarity > max_similarity:
                        max_similarity = similarity
            if max_similarity >= threshold:
                alignments.append((token1, token2, max_similarity))

    return alignments, tokens1, tokens2

def join_n_grams(sentence,n):
    tokens = word_tokenize(sentence)
    n_grams = list(nltk.ngrams(tokens, n))
    if n == 1:
        n_grams = [gram[0] for gram in n_grams]
    else:
        n_grams = ['_'.join(gram) for gram in n_grams]
    return n_grams

def get_alignments(sentence1, sentence2,n):

    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()

    sentence1 = contractions.fix(sentence1)
    sentence2 = contractions.fix(sentence2)

    sent1 = nlp(sentence1)
    sent2 = nlp(sentence2)
 
    lemmatized_tokens1 = [token.lemma_ for token in sent1]
    lemmatized_tokens2 = [token.lemma_ for token in sent2]
 
    lemmatized_sentence1 = ' '.join(lemmatized_tokens1)
    lemmatized_sentence2 = ' '.join(lemmatized_tokens2)

    alignments, tokens1, tokens2 = align_sentences(lemmatized_sentence1, lemmatized_sentence2,n)

    tokens1 = [token for token in tokens1 if token.lower() not in nltk.corpus.stopwords.words('english') and token not in ['.', ',', '?', '!', ':', ';']]
    tokens2 = [token for token in tokens2 if token.lower() not in nltk.corpus.stopwords.words('english') and token not in ['.', ',', '?', '!', ':', ';']]

    alignments = sorted(alignments, key=lambda x: x[2], reverse=True)

    used_tokens1_set = set()
    used_tokens2_set = set()

    final_alignments = []

    for alignment in alignments:
        token1, token2, similarity = alignment
        token1_split = token1.split('_')
        token2_split = token2.split('_')

        if token1 in used_tokens1_set or token2 in used_tokens2_set:
            continue
        if any([token in used_tokens1_set for token in token1_split]):
            continue
        if any([token in used_tokens2_set for token in token2_split]):
            continue
        final_alignments.append((token1, token2, similarity))

        used_tokens1_set.add(token1)
        used_tokens2_set.add(token2)

        for token in token1_split:
            used_tokens1_set.add(token)
        for token in token2_split:
            used_tokens2_set.add(token)

    unigram_counts1 = len([token for token in tokens1 if '_' not in token])
    unigram_counts2 = len([token for token in tokens2 if '_' not in token])

    return final_alignments, unigram_counts1, unigram_counts2

if __name__ == "__main__":
    sentence1 = "She began her journey in the early morning."
    sentence2 = "She started her trip at dawn."
    alignments, unigram_counts1, unigram_counts2 = get_alignments(sentence1, sentence2, 3)
    print(alignments)