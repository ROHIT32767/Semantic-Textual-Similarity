import pandas as pd
import contractions
import nltk
import numpy as np

def preprocess():
    data = pd.read_csv(f'../../train.csv')
    sentence1 = data['s1'].values
    sentence2 = data['s2'].values
    labels = data['score'].values

    sentence1 = [s.lower() for s in sentence1]
    sentence1 = [contractions.fix(s) for s in sentence1]
    sentence1 = [nltk.word_tokenize(s) for s in sentence1]
    sentence1 = [[w for w in s if w.isalnum()] for s in sentence1]

    sentence2 = [s.lower() for s in sentence2]
    sentence2 = [contractions.fix(s) for s in sentence2]
    sentence2 = [nltk.word_tokenize(s) for s in sentence2]
    sentence2 = [[w for w in s if w.isalnum()] for s in sentence2]

    embeddings_index = {}
    with open('../glove_embedings/glove.840B.300d.txt') as f:
        for line in f:
            values = line.split()
            coefs = np.asarray(values[-300:], dtype='float32')
            word = ' '.join(values[:-300])
            embeddings_index[word] = coefs

    embeddings_index['UNK'] = np.zeros(300)

    embeddings_index['PAD'] = np.zeros(300)

    # lengths = [len(s) for s in sentence1] + [len(s) for s in sentence2]
    # max_len = np.percentile(lengths, 95)

    # max_len = int(max_len)

    max_len = 30

    print("Max length of sentence: ", max_len)

    pos_tags = ['UNK','CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBG','VBD','VBN','VBP','VBZ','WDT','WP','WRB']

    sent1 = []
    sent2 = []

    for s1, s2 in zip(sentence1, sentence2):
        s1_original = s1.copy()
        s2_original = s2.copy()
        s1_tagged = nltk.pos_tag(s1)
        s2_tagged = nltk.pos_tag(s2)
        for i in range(max_len):
            if i < len(s1):
                if s1[i] in embeddings_index:
                    s1[i] = embeddings_index[s1[i]]
                    if s1_original[i] in s2_original:
                        s1[i] = np.concatenate((s1[i], [1]))
                    else:
                        s1[i] = np.concatenate((s1[i], [0]))
                    one_hot = np.zeros(36)
                    if s1_tagged[i][1] in pos_tags:
                        one_hot[pos_tags.index(s1_tagged[i][1])] = 1
                    else:
                        one_hot[pos_tags.index('UNK')] = 1
                    s1[i] = np.concatenate((s1[i], one_hot)) 
                else:
                    s1[i] = embeddings_index['UNK']
                    s1[i] = np.concatenate((s1[i], [0]))
                    one_hot = np.zeros(36)
                    one_hot[pos_tags.index('UNK')] = 1
                    s1[i] = np.concatenate((s1[i], one_hot))
            else:
                s1.append(embeddings_index['PAD'])
                s1[i] = np.concatenate((s1[i], [0]))
                one_hot = np.zeros(36)
                one_hot[pos_tags.index('UNK')] = 1
                s1[i] = np.concatenate((s1[i], one_hot))

            if i < len(s2):
                if s2[i] in embeddings_index:
                    s2[i] = embeddings_index[s2[i]]
                    if s2_original[i] in s1_original:
                        s2[i] = np.concatenate((s2[i], [1]))
                    else:
                        s2[i] = np.concatenate((s2[i], [0]))
                    one_hot = np.zeros(36)
                    if s2_tagged[i][1] in pos_tags:
                        one_hot[pos_tags.index(s2_tagged[i][1])] = 1
                    else:
                        one_hot[pos_tags.index('UNK')] = 1
                    s2[i] = np.concatenate((s2[i], one_hot))
                else:
                    s2[i] = embeddings_index['UNK']
                    s2[i] = np.concatenate((s2[i], [0]))
                    one_hot = np.zeros(36)
                    one_hot[pos_tags.index('UNK')] = 1
                    s2[i] = np.concatenate((s2[i], one_hot))
            else:
                s2.append(embeddings_index['PAD'])
                s2[i] = np.concatenate((s2[i], [0]))
                one_hot = np.zeros(36)
                one_hot[pos_tags.index('UNK')] = 1
                s2[i] = np.concatenate((s2[i], one_hot))

        sent1.append(s1[:max_len])
        sent2.append(s2[:max_len])

    sentence1 = np.array(sent1)
    sentence2 = np.array(sent2)

    labels = np.array(labels)

    return sentence1, sentence2, labels

if __name__ == '__main__':
    preprocess()