import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

from math import sqrt

import os

Window_size=16
num_steps =1

def lstm_data_transform(x_data, y_data, num_steps=Window_size):
    x_array = np.array([x_data[i:i+num_steps] for i in range(x_data.shape[0]-num_steps)])
    y_array = y_data[num_steps:]
    return x_array, y_array

def windowsfun(df, i, num_steps):
    x_data = df.iloc[:, i].values
    X = [x_data[i:end_ix].flatten() for i, end_ix in enumerate(range(num_steps, df.shape[0])) if end_ix < df.shape[0]]
    return pd.DataFrame(X)

def FinalDf(input_df, num_steps):
    windowed_data = []
    x_df = input_df.drop(['GPS_VxF'], axis=1)
    for i in range(len(x_df.columns)):
        windowed_feature = windowsfun(x_df, i, num_steps)
        windowed_data.append(windowed_feature)
    updated_df = pd.concat(windowed_data, axis=1)
    updated_df.columns = range(len(updated_df.columns))
    updated_df['GPS_VxF'] = input_df['GPS_VxF'].values[num_steps:]
    return updated_df
def sliceddf(df):
    part=[[.2,.5],
        [.3,.7],
        [.7,.95],
        [.11,.35],
        [.45,.85],
        [.8,.99],
        [.35,.65],
        [.25,.75],
        [.6,.75]]
    part=np.array(part)
    part=part*(len(df))
    part=part.astype(int)
    df1 = pd.DataFrame(columns=['Vwhl_FL', 'Vwhl_FR', 'Vwhl_RL', 'Vwhl_RR'])
    for x in part:
        df1=pd.concat([df1,df.iloc[x[0]:x[1]]])
    return df1
Train_path="/home/aesicd_42/Desktop/tejas/Hyundai_project/DATA/curve__circle__snow_ice/Ice"


def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

files= getListOfFiles(Train_path)
Cons_X = list()
cons_y = list()

files_xls = [f for f in files if f[-4:] == '.csv']
df = pd.DataFrame()
df_new =pd.DataFrame()
for f in files_xls:

    data = pd.read_csv(f)
    # data1=sliceddf(data)

    # df =pd.concat([data1,data])
    
    df=data
    
    MAX_Vwhl_F = df[['Vwhl_FL', 'Vwhl_FR','Vwhl_RL','Vwhl_RR']].max(axis=1)
    df['MAX_Vwhl'] = MAX_Vwhl_F
    
    MIN_Vwhl_R = df[['Vwhl_FL', 'Vwhl_FR','Vwhl_RL','Vwhl_RR']].min(axis=1)
    df['MIN_Vwhl'] = MIN_Vwhl_R

    MAX_MIN = df['MAX_Vwhl']/df['MIN_Vwhl']
    df['MAX_MIN'] = MAX_MIN
    
    MAX_MIN_Diff=df['MAX_Vwhl']-df['MIN_Vwhl']
    df['MAX_MIN_Diff'] = MAX_MIN_Diff
    
    Rear_Mean_Vwhl = df[['Vwhl_RL', 'Vwhl_RR']].mean(axis=1)
    df['Rear_Mean_Vwhl'] = Rear_Mean_Vwhl
    
    Front_Max_Vwhl = df[['Vwhl_FL', 'Vwhl_FR']].mean(axis=1)
    df['Front_Max_Vwhl'] = Front_Max_Vwhl

    df['Vwhl_FL_diff'] = df['Vwhl_FL'].diff()
    df['Vwhl_FR_diff'] = df['Vwhl_FR'].diff()
    df['Vwhl_RL_diff'] = df['Vwhl_RL'].diff()
    df['Vwhl_RR_diff'] = df['Vwhl_RR'].diff()
    
    df['Vwhl_FR_w'] = 0.0
    df['Vwhl_FL_w'] = 0.0
    df['Vwhl_RL_w'] = 0.0
    df['Vwhl_RR_w'] = 0.0
    
    def sigmoid(x, scale=.1):
        return 1 / (1 + np.exp(-scale * x))

    # Apply sigmoid function to 'YawRate'
    df['sigmoid_yaw_rate'] = sigmoid(df['YawRate'])
    wfp=1.1
    wfn=0.9
    
    df['Vwhl_FR_w'] = np.where(df['YawRate'] > 0, df['Vwhl_FR'] + (1 + df['sigmoid_yaw_rate']) * wfp, df['Vwhl_FR'] * wfn)
    df['Vwhl_FL_w'] = np.where(df['YawRate'] < 0, df['Vwhl_FL'] + (-1 + df['sigmoid_yaw_rate']) * wfp, df['Vwhl_FL'] * wfn)
    df['Vwhl_RR_w'] = np.where(df['YawRate'] > 0, df['Vwhl_RR'] + (1 + df['sigmoid_yaw_rate']) * wfp, df['Vwhl_RR'] * wfn)
    df['Vwhl_RL_w'] = np.where(df['YawRate'] < 0, df['Vwhl_RL'] + (-1 + df['sigmoid_yaw_rate']) * wfp, df['Vwhl_RR'] * wfn)

    # df = df[['Vwhl_FL','Vwhl_FL_diff','Vwhl_FR', 'Vwhl_FR_diff','Vwhl_RL', 'Vwhl_RL_diff', 'Vwhl_RR','Vwhl_RR_diff', 'MAX_MIN','Rear_Mean_Vwhl','GPS_VxF']]
    df = df[['Vwhl_FL_w','Vwhl_FR_w', 'Vwhl_RL_w', 'Vwhl_RR_w','Rear_Mean_Vwhl', 'Front_Max_Vwhl','MAX_MIN','YawRate','GPS_VxF']]
    df.dropna(inplace=True)


    Filtered_df= FinalDf(df,num_steps)
    x=Filtered_df.drop(['GPS_VxF'],axis=1)
    y =Filtered_df[['GPS_VxF']]

    x_new, y_new = lstm_data_transform(x, y, num_steps=Window_size)
    Cons_X.append(x_new)
    cons_y.append(y_new)
  
# Make final arrays
x_array = np.array(Cons_X,dtype=object)
y_array = np.array(cons_y,dtype=object)


X_Train_Tensor = x_array[0]
Y_Train_Tensor = y_array[0]

for k in range(1,len(x_array)):
    X_Train_Tensor=np.concatenate((X_Train_Tensor, x_array[k]))
    Y_Train_Tensor=np.concatenate((Y_Train_Tensor, y_array[k]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import train_test_split
import datetime
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers, hidden_size, dropout, max_length=100):
        super(TransformerModel, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size//2, kernel_size=3)
        self.conv1d_2 = nn.Conv1d(in_channels=hidden_size//2, out_channels=hidden_size, kernel_size=3)
        self.positional_encoding = PositionalEncoding(hidden_size, max_length)
        
        self.self_attn_layers = nn.ModuleList([
            nn.Sequential(
                LayerNorm(hidden_size),
                MultiHeadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                LayerNorm(hidden_size),
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size*2),
                    nn.ReLU(),
                    nn.Linear(hidden_size*2, hidden_size)
                )
            )
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # transpose to (batch_size, input_size, seq_length)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = x.transpose(1, 2)  # transpose back to (batch_size, seq_length, hidden_size)
        x = self.positional_encoding(x)
        
        for self_attn, feed_forward in zip(self.self_attn_layers, self.feed_forward_layers):
            x = x + self_attn(x)[0]
            x = x + feed_forward(x)
        x = torch.mean(x, dim=1)
        x.requires_grad_(True)  # Set requires_grad to True
        x = self.fc(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        query = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softsign(scores)  # Replaced softmax with softsign
        attn_probs = self.dropout(attn_probs)
        weighted_values = torch.matmul(attn_probs, value).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        x = self.out_proj(weighted_values)
        return x,


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.features = features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.weight * normalized + self.bias
    
class CustomLoss(nn.Module):
    def __init__(self, penalty_weight=2.0):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        absolute_error = torch.abs(predictions - targets)
        count = torch.count_nonzero(absolute_error >= 2.5)
        per = count / len(targets)
        penalty = torch.where(absolute_error >= 1.5, (absolute_error - 1.5) * self.penalty_weight, torch.zeros_like(absolute_error))
        loss = mse/2 + penalty.mean() + per*4
        return loss
    
# Set hyperparameters
NUMBER_HEADS=2
INPUT_SIZE = 8
OUTPUT_SIZE = 1
NUM_LAYERS = 2
HIDDEN_SIZE = 32
DROPOUT = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 8192*2*2*4
NUM_EPOCHS = 500

model = TransformerModel(INPUT_SIZE, OUTPUT_SIZE, NUMBER_HEADS,NUM_LAYERS, HIDDEN_SIZE, DROPOUT).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = CustomLoss()
# Create data loaders
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Train the model
from tqdm import tqdm

import csv

train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model_path = 'best_model_trial_06.pth'

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}: train", leave=False) as pbar:
        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss /= len(train_loader.dataset)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        with tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}: test", leave=False) as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            test_loss /= len(test_loader.dataset)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), best_model_path)
        
filename = "losses_1.csv"  # Change the filename as desired

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Test Loss"])  # Write header
    for epoch in range(NUM_EPOCHS):
        writer.writerow([epoch+1, train_losses[epoch], test_losses[epoch]])