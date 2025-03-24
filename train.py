import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import FusionModel

datasetpath = r'F:\eyes_projects\tiny_dataset'

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize components
    model = FusionModel().to(device)
    train_loader = get_dataloader(datasetpath)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            *inputs, labels = batch
            inputs = [x.to(device) for x in inputs]
            labels = labels.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
        
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f}')

if __name__ == '__main__':
    train()