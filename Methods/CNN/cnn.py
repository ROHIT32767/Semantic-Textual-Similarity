import numpy as np
import pandas as pd
import torch
from cnn_utils import preprocess, Model, train_model, load_data
import torch.utils.data as data_utils
import sys

def main():
    X1_train, X2_train, y_train, X1_val, X2_val, y_val,_,_,_ = load_data()
    print(f"X1_train shape: {X1_train.shape}")
    in_channels = 1
    out_channels = 300
    kernel_size = 337

    fc_input_size = 2 * out_channels
    fc_hidden_size = 300
    fc_output_size = 1
    fc_activation = 'tanh'

    train = data_utils.TensorDataset(X1_train, X2_train, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=32, shuffle=True)

    val = data_utils.TensorDataset(X1_val, X2_val, y_val)
    val_loader = data_utils.DataLoader(val, batch_size=32, shuffle=True)

    model = Model(in_channels, out_channels, kernel_size, fc_input_size, fc_hidden_size, fc_output_size, fc_activation)
    train_loss, val_loss, model = train_model(model, train_loader, val_loader, epochs=10, lr=0.001)
    torch.save(model, f'models/model.pt')

if __name__ == '__main__':
    main()

