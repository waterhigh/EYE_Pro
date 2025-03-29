# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Data import create_loaders
from model import OcularAgeModel


class TrainingConfig:
    def __init__(self):
        self.seed = 42                
        self.epochs = 50             
        self.lr = 1e-4                
        self.batch_size = 32           
        self.save_dir = "checkpoints" 
        self.early_stop = 10           
        self.verbose = True           

def setup_environment(config):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA  is invalid")
        
 
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    device = torch.device("cuda")
    print(f" device: {torch.cuda.get_device_name(0)}")
    return device

def initialize_model(device):
 
    model = OcularAgeModel().to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    criterion = nn.L1Loss()  
    
    return model, optimizer, scheduler, criterion

def compute_accuracy(predictions, targets):

    abs_errors = torch.abs(predictions - targets)
    accuracies = {
        '1yr': (abs_errors <= 1.0).float().mean().item(),
        '5yr': (abs_errors <= 5.0).float().mean().item(),
        '10yr': (abs_errors <= 10.0).float().mean().item()
    }
    return accuracies

def train_epoch(model, loader, device, optimizer, criterion, scaler):
 
    model.train()
    total_loss = 0.0
    acc_1yr = 0.0
    acc_5yr = 0.0
    acc_10yr = 0.0
    
    for batch_idx, batch in enumerate(loader):
    
        eyes = batch['eye'].to(device, non_blocking=True)
        pupils = batch['pupil'].to(device, non_blocking=True)
        ages = batch['age'].to(device, non_blocking=True)
        
    
        optimizer.zero_grad(set_to_none=True)
        
      
        with autocast():
            pred_ages = model(eyes, pupils)
            loss = criterion(pred_ages, ages)
        
 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
 
        batch_size = eyes.size(0)
        total_loss += loss.item() * batch_size
        
        acc = compute_accuracy(pred_ages.detach(), ages)
        acc_1yr += acc['1yr'] * batch_size
        acc_5yr += acc['5yr'] * batch_size
        acc_10yr += acc['10yr'] * batch_size
        

    num_samples = len(loader.dataset)
    avg_loss = total_loss / num_samples
    avg_acc = {
        '1yr': acc_1yr / num_samples,
        '5yr': acc_5yr / num_samples,
        '10yr': acc_10yr / num_samples
    }
    
    return avg_loss, avg_acc

def validate(model, loader, device, criterion):

    model.eval()
    total_loss = 0.0
    acc_1yr = 0.0
    acc_5yr = 0.0
    acc_10yr = 0.0
    
    with torch.no_grad():
        for batch in loader:
            eyes = batch['eye'].to(device)
            pupils = batch['pupil'].to(device)
            ages = batch['age'].to(device)
            
            pred_ages = model(eyes, pupils)
            loss = criterion(pred_ages, ages)
            
            batch_size = eyes.size(0)
            total_loss += loss.item() * batch_size
            
            acc = compute_accuracy(pred_ages, ages)
            acc_1yr += acc['1yr'] * batch_size
            acc_5yr += acc['5yr'] * batch_size
            acc_10yr += acc['10yr'] * batch_size
    
    num_samples = len(loader.dataset)
    avg_loss = total_loss / num_samples
    avg_acc = {
        '1yr': acc_1yr / num_samples,
        '5yr': acc_5yr / num_samples,
        '10yr': acc_10yr / num_samples
    }
    
    return avg_loss, avg_acc

def save_checkpoint(model, optimizer, epoch, loss, save_path):

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': loss,
    }, save_path)

def main():
 
    config = TrainingConfig()
    
    try:
   
        device = setup_environment(config)
        
   
        os.makedirs(config.save_dir, exist_ok=True)
        
 
        model, optimizer, scheduler, criterion = initialize_model(device)
        scaler = GradScaler()
        
  
        loaders = create_loaders(batch_size=config.batch_size)
        

        best_loss = float('inf')
        epochs_no_improve = 0
        train_history = []
        
    
        for epoch in range(config.epochs):
            start_time = time.time()
            
      
            train_loss, train_acc = train_epoch(
                model, loaders['train'], device, optimizer, criterion, scaler
            )
            
      
            val_loss, val_acc = validate(
                model, loaders['val'], device, criterion
            )
            
       
            scheduler.step(val_loss)
            
      
            train_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr']
            })
            
     
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(
                    model, optimizer, epoch, best_loss,
                    os.path.join(config.save_dir, 'best_model.pth')
                )
            else:
                epochs_no_improve += 1
                
       
            if epochs_no_improve >= config.early_stop:
                print("early stop")
                break
            
 
            epoch_time = time.time() - start_time
            if config.verbose:
                print(f"\nEpoch {epoch+1}/{config.epochs} ({epoch_time:.1f}s)")
                print(f"Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")
                print(f"Train Acc: <1 year: {train_acc['1yr']*100:.1f}% | <5 years: {train_acc['5yr']*100:.1f}% | <10 years: {train_acc['10yr']*100:.1f}%")
                print(f"Val Acc:   <1 year: {val_acc['1yr']*100:.1f}% | <5 years: {val_acc['5yr']*100:.1f}% | <10 years: {val_acc['10yr']*100:.1f}%")
                print(f"now lr: {optimizer.param_groups[0]['lr']:.1e}")
        
 
        print("\n test the best weight...")
        model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_model.pth'))['model_state_dict'])
        test_loss, test_acc = validate(model, loaders['test'], device, criterion)
        
        print(f"\n test result:")
        print(f"test loss: {test_loss:.2f}")
        print(f"test acc: <1 year: {test_acc['1yr']*100:.1f}% | <5 years: {test_acc['5yr']*100:.1f}% | <10 years: {test_acc['10yr']*100:.1f}%")
    
    except Exception as e:
        print(f"\n error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
