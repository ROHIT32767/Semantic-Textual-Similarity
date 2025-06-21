import contractions
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


def preprocess(file):

    # SICK dataset
    if file == 'sts':
        data = pd.read_csv('../data/stsbenchmark/train.csv')
        sentence1 = data['sentence_A'].values
        sentence2 = data['sentence_B'].values
        labels = data['normalised_score'].values

    # STS Benchmark dataset
    if file == 'sick':
        data = pd.read_csv('../data/sick/train.csv')
        sentence1 = data['sentence1'].values
        sentence2 = data['sentence2'].values
        labels = data['similarity'].values

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

def load_data():
    directory = '../utils/data/'
    
    X1 = torch.load(directory + 's1_train.pt', weights_only=False)
    X2 = torch.load(directory + 's2_train.pt', weights_only=False)
    y = torch.load(directory + 'y_train.pt', weights_only=False)
    X1_val = torch.load(directory + 's1_val.pt', weights_only=False)
    X2_val = torch.load(directory + 's2_val.pt', weights_only=False)
    y_val = torch.load(directory + 'y_val.pt', weights_only=False)
    X1_train = torch.Tensor(X1).float()
    X2_train = torch.Tensor(X2).float()
    y_train = torch.Tensor(y).float()
    X1_val = torch.Tensor(X1_val).float()
    X2_val = torch.Tensor(X2_val).float()
    y_val = torch.Tensor(y_val).float()
    X1_test = torch.load(directory + 's1_test.pt', weights_only=False)
    X2_test = torch.load(directory + 's2_test.pt', weights_only=False)
    y_test = torch.load(directory + 'y_test.pt', weights_only=False)
    X1_test = torch.Tensor(X1_test).float()
    X2_test = torch.Tensor(X2_test).float()
    y_test = torch.Tensor(y_test).float()
    return X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test
    

# CNN architecture
# no.of conv layers = 1
# no.of filters = 300
# kernel length = 337
# activation function = ReLU
# 1-D kernel
# input shape = (batch_size, 337)
# output shape = (batch_size, 300)

class CNN(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=inchannels, out_channels=outchannels, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = x.squeeze(2)
        return x
    

# Max pooling layer
# input shape = (batch_size, 22, 300)
# output shape = (batch_size, 300)

class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()

    def forward(self, x):
        x, _ = torch.max(x, dim=1)
        return x
    
# Fully connected Neural Network
# input shape = (batch_size, 600)
# output shape = (batch_size, 1)
# no.of hidden layers = 1
# no.of neurons in hidden layer = 300
# first hidden layer activation function = tanh
# output layer activation function = linear

class FCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='tanh')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
# Final model
# input shape = (batch_size, 22, 337)
# output shape = (batch_size, 1)
# each word is represented by 337 features is passed through a CNN layer => output shape = (batch_size, 300)
# 22 words are passed through the CNN layer => 22 outputs of shape (batch_size, 300)
# then 22 outputs are passed through a max pooling layer(batch_size, 22, 300) => output shape = (batch_size, 300)
# there are 2 sentences => 2 outputs of shape (batch_size, 300)
# element wise subtraction of 2 outputs => output shape = (batch_size, 300)
# element wise multiplication of 2 outputs => output shape = (batch_size, 300)
# concatenate the 2 outputs => output shape = (batch_size, 600)
# pass the output through a fully connected neural network => output shape = (batch_size, 1)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fc_input_size, fc_hidden_size, fc_output_size, fc_activation):
        super(Model, self).__init__()
        self.cnn = CNN( in_channels, out_channels, kernel_size)
        self.maxpool = MaxPool()
        self.fcnn = FCNN( fc_input_size, fc_hidden_size, fc_output_size, fc_activation)

    def forward(self, x1, x2):
        # apply CNN on each word in sentence1
        out_words_1 = [self.cnn(x1[:,i,:]) for i in range(x1.size(1))]
        # out_words_1 = [batch_size, 300]
        out_words_1 = torch.stack(out_words_1, dim=1)
        # out_words_1 = [batch_size, 22, 300]
        out_words_1 = self.maxpool(out_words_1)
        # out_words_1 = [batch_size, 300]

        # apply CNN on each word in sentence2
        out_words_2 = [self.cnn(x2[:,i,:]) for i in range(x2.size(1))]
        # out_words_2 = [batch_size, 300]
        out_words_2 = torch.stack(out_words_2, dim=1)
        # out_words_2 = [batch_size, 22, 300]
        out_words_2 = self.maxpool(out_words_2)
        # out_words_2 = [batch_size, 300]

        # element wise subtraction
        out_sub = out_words_1 - out_words_2
        # out_sub = [batch_size, 300]

        # element wise multiplication
        out_mul = out_words_1 * out_words_2
        # out_mul = [batch_size, 300]

        # concatenate
        out = torch.cat((out_sub, out_mul), dim=1)
        # out = [batch_size, 600]

        out = self.fcnn(out)
        # out = [batch_size, 1]

        return out
    
def train_model(model, train_loader, val_loader,epochs = 5, lr = 0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = len(train_loader)
    val_batches = len(val_loader)

    loss_train = []
    loss_val = []


    for epoch in tqdm(range(epochs), desc='Epochs'):
        model.train()
        running_loss = 0.0
        for i,data in tqdm(enumerate(train_loader), leave=False, desc='Batches', total=n_batches):
            x1, x2, target = data
            optimizer.zero_grad()
            output = model(x1, x2)
            loss = criterion(output, target.view(-1,1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss_train.append(running_loss/n_batches)
        print("Epoch: {} Train Loss: {}".format(epoch+1, running_loss/n_batches))

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for i,data in enumerate(val_loader):
                x1, x2, target = data
                output = model(x1, x2)
                loss = criterion(output, target.view(-1,1).float())
                val_running_loss += loss.item()

        loss_val.append(val_running_loss/val_batches)
        print("Epoch: {} Validation Loss: {}".format(epoch+1, val_running_loss/val_batches))

    return loss_train, loss_val, model
