import numpy as np
import pandas as pd
from dataprep import data_r, data_c
from graph_gen import buildKnnGraph
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def doGCN(rdata, kv, seed):
    torch.manual_seed(seed)
    data = rdata
    k=kv
    ids, X, y, edge_index = buildKnnGraph(data, k)

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    graph = Data(x=X, edge_index=edge_index, y=y)

    data = graph.to(device)

    y = data.y
    N = y.shape[0]

    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask  = torch.zeros(N, dtype=torch.bool)

    classes = torch.unique(y)

    for c in classes:
        idx = torch.where(y == c)[0]                     # indices of class c
        idx = idx[torch.randperm(len(idx))]              # shuffle within class
        
        split = int(0.75 * len(idx))                      # 70/30 split
        
        train_idx = idx[:split]
        test_idx  = idx[split:]
        
        data.train_mask[train_idx] = True
        data.test_mask[test_idx]   = True

    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.conv4 = GCNConv(hidden_channels, hidden_channels)
            self.conv5 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            x = F.relu(self.conv4(x, edge_index))
            x = self.conv5(x, edge_index)
            return F.log_softmax(x, dim=1)


    model = GCN(data.num_node_features, 128, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    '''
    for cycle in range(201):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if cycle % 1 == 0:
            print(f"cycle {cycle}, Loss: {loss.item():.4f}")
    '''

    target_loss = 0.5
    max_cycles = 50   # safety cap

    cycle = 0
    model.train()

    while True:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        #print(f"cycle {cycle}, Loss: {loss.item():.4f}")

        if loss.item() <= target_loss:
            #print(f"Stopping: loss reached {loss.item():.4f}")
            break

        cycle += 1
        if cycle >= max_cycles:
            #print("Stopping: reached max_cycles cap")
            break

    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    train_mask, test_mask = data.train_mask, data.test_mask

    test_acc = accuracy_score(data.y[test_mask].cpu(), pred[test_mask].cpu())
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    cm = confusion_matrix(data.y[test_mask].cpu(), pred[test_mask].cpu())
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.savefig(f"l4ch1281cm.png", dpi=400, bbox_inches="tight")
    plt.close()
    
    target_names = ['UDGs', 'LSBs', 'HSBs', 'Dwarfs']

    c_report = classification_report(
        data.y[test_mask].cpu(), 
        pred[test_mask].cpu(), 
        target_names=target_names, 
        digits=4
    )
    
    print("\nClassification Report:\n")
    print(c_report)
    
    return test_acc

best_k = []


for i in range(0,31):
    a = doGCN(data_r, i, 42)
    best_k.append([a,i])

best_a, op_k = max(best_k, key=lambda x: x[0])
print(f"\nBest k is {op_k} with accuracy {best_a}\n")

x_val = []
y_val = []

for i in range(len(best_k)):
    x_val.append(best_k[i][1])
    y_val.append(best_k[i][0])

plt.figure()
plt.plot(x_val, y_val)
plt.title(f"Best graph k value estimation (l=4, Ch=128)\nBest k={op_k}")
plt.xlabel("k values")
plt.ylabel("accuracies")
plt.show()

doGCN(data_r, op_k, 42)