# src/utils.py
import torch
import matplotlib.pyplot as plt
import os

def save_model(model, optimizer, epoch, loss, path):
    """
    实现模型保存功能 
    """
    print(f"Saving model checkpoint to {path}")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model(model, optimizer, path):
    """
    实现模型加载功能 
    """
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}, starting from scratch.")
        return model, optimizer, 0, None
        
    print(f"Loading model checkpoint from {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss

def plot_curves(train_losses, val_losses, plot_path):
    """
    实现训练曲线可视化并保存到 results/ [cite: 15, 19]
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
        
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")

def count_parameters(model):
    """
    实现参数统计功能 
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)