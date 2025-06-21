import torch
import torch.nn as nn
from rnn_utils import load_data, Model, train
import sys

def main():
    X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = load_data()
    rnn_input_size = 337
    rnn_hidden_size = 64
    rnn_num_layers = 2
    rnn_output_size = 300
    rnn_activation = 'relu'
    fc_input_size = 2 * rnn_output_size
    fc_hidden_size = 128
    fc_output_size = 1
    fc_activation = 'tanh'
    fc_num_layers = 1
    criterion = nn.MSELoss()
    lr = 0.001
    n_epochs = 100
    batch_size = 32

    model = Model(rnn_input_size, rnn_hidden_size, rnn_num_layers, rnn_output_size, rnn_activation, fc_input_size, fc_hidden_size, fc_output_size, fc_activation, fc_num_layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_losses, val_losses = train(model, criterion, X1_train, X2_train, y_train, X1_val, X2_val, y_val, n_epochs, batch_size, device, lr)

    torch.save(model, f'models/model.pt')

if __name__ == '__main__':
    main()