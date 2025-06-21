import torch
import torch.nn as nn
from lstm_utils import Model, train, load_data
import sys

def main():
    X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = load_data()
    lstm_input_size = 337
    lstm_hidden_size = 128
    lstm_num_layers = 1
    lstm_output_size = 300
    lstm_activation = 'relu'
    lstm_bidirectional = True
    fc_input_size = 2 * lstm_output_size
    fc_hidden_size = 128
    fc_output_size = 1
    fc_activation = 'relu'
    fc_num_layers = 2
    lr = 0.001
    n_epochs = 100
    batch_size = 16

    model = Model(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_output_size, lstm_activation, lstm_bidirectional, fc_input_size, fc_hidden_size, fc_output_size, fc_activation, fc_num_layers)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_losses, val_losses = train(model, criterion, X1_train, X2_train, y_train, X1_val, X2_val, y_val, n_epochs, batch_size, device, lr)

    torch.save(model, f'models/model.pt')

if __name__ == '__main__':
    main()