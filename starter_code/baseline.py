import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn . model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  
        x = self.lin(x)
        return x

dataset = pd.read_parquet(r'..\data\train.parquet')
train, val = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

graph_list = []

for _, row in train.iterrows():
    x = torch.tensor(np.vstack(row['node_feat']).astype(np.float32))
    edge_index = torch.tensor(np.vstack(row['edge_index']).astype(np.int32))
    edge_attr = torch.tensor(np.vstack(row['edge_attr']).astype(np.float32)) 
    y = torch.tensor(row['y'], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    graph_list.append(data)
    
train_loader = DataLoader(graph_list, batch_size=32, shuffle=False)

num_node_features = 38 
num_classes = 2  

model = GCN(num_node_features, hidden_channels=64, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(300):  
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    
val_graph_list = []

for _, row in val.iterrows():
    x = torch.tensor(np.vstack(row['node_feat']).astype(np.float32))
    edge_index = torch.tensor(np.vstack(row['edge_index']).astype(np.int64))
    edge_attr = torch.tensor(np.vstack(row['edge_attr']).astype(np.float32))
    y = torch.tensor(row['y'], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    val_graph_list.append(data)

val_loader = DataLoader(val_graph_list, batch_size=32, shuffle=False)

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in val_loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        
    score = f1_score(y_true, y_pred, average ='macro')
    
print (f'Validation F1 Score : {score:.4f}')

test = pd.read_parquet(r'..\data\test.parquet')

test_graph_list = []

for _, row in test.iterrows():
    x = torch.tensor(np.vstack(row['node_feat']).astype(np.float32))
    edge_index = torch.tensor(np.vstack(row['edge_index']).astype(np.int64))
    edge_attr = torch.tensor(np.vstack(row['edge_attr']).astype(np.float32))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    test_graph_list.append(data)

test_loader = DataLoader(test_graph_list, batch_size=32, shuffle=False)

test_preds =[]
with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        test_preds.extend(pred.cpu().numpy())
        
pd.DataFrame({'graph_id': test ['graph_id'], 'y_pred': test_preds}).to_parquet('../submissions/sample_submission.parquet', index=False)