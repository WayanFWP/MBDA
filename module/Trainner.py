import torch
import torch.nn as nn

import numpy as np

class Trainner:
    def __init__(self, model, optimizer="adam", lr=0.001, criterion=nn.MSELoss(), flatten = True):
        self.model = model
        self.flatten = flatten
        
        self.lr = lr
        self.criterion = criterion
        
        _optims = {
            'adam':    torch.optim.Adam,
            'sgd':     torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
            'adamw':   torch.optim.AdamW,
        }
        if optimizer not in _optims:
            raise ValueError(f"Optimizer '{optimizer}' not supported. Choose from: {list(_optims.keys())}")
        
        self.optimizer = _optims[optimizer](self.model.parameters(), lr=lr)
        
        self.loss_history = []
        self.accuracy_history = []
                
    def train_steps(self, X, y):
        self.optimizer.zero_grad()
        
        output = self.model(X)
        
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), (output.argmax(dim=1) == y).sum().item(), y.size(0)
    
    def fit(self, data, epochs, verbose=True, val_data=None):
        self.val_loss_history = []
        self.val_accuracy_history = []
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss, correct, total = 0, 0, 0
            for X, y in data:
                X = self.prepareInput(X)
                curr_loss, curr_correct, curr_total = self.train_steps(X, y)
                epoch_loss += curr_loss
                correct += curr_correct
                total += curr_total

            self.loss_history.append(epoch_loss / len(data))
            self.accuracy_history.append(100 * correct / total)

            if val_data is not None:
                val_loss, val_correct, val_total = 0, 0, 0
                self.model.eval()
                with torch.no_grad():
                    for X, y in val_data:
                        X = self.prepareInput(X)
                        out = self.model(X)
                        val_loss += self.criterion(out, y).item()
                        val_correct += (out.argmax(dim=1) == y).sum().item()
                        val_total += y.size(0)
                self.val_loss_history.append(val_loss / len(val_data))
                self.val_accuracy_history.append(100 * val_correct / val_total)
                self.model.train()

            if verbose:
                val_info = f" - Val Loss: {self.val_loss_history[-1]:.4f} - Val Acc: {self.val_accuracy_history[-1]:.4f}" if val_data else ""
                print(f"Epoch {epoch+1}/{epochs} - Loss: {self.loss_history[-1]:.4f} - Acc: {self.accuracy_history[-1]:.4f}{val_info}")

            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < 1e-3:
                if verbose: print("Early stopping triggered.")
                break
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            return torch.argmax(output, dim=1)
        
    def evaluate(self, data):
        concate_preds, concate_labels, concate_image= [], [], []
        
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in data:
                X = self.prepareInput(X)  # Flatten images
                output = self.model(X)
                pred = self.predict(X)
                correct += (output.argmax(dim=1) == y).sum().item()
                
                concate_preds.extend(pred.cpu().numpy())
                concate_labels.extend(y.cpu().numpy())
                concate_image.extend(X.cpu().numpy())
                total += y.size(0)  
                
            concate_preds = np.array(concate_preds)
            concate_labels = np.array(concate_labels)
            concate_image = np.array(concate_image)         
            
        return 100 * (correct / total), concate_preds, concate_labels, concate_image
    
    def prepareInput(self, X):
        return X.view(X.size(0), -1) if self.flatten else X

    # Save for Inference only
    def save(self, path):
        arch_type = "ANN" if self.flatten else "CNN"
        torch.save({
            "arch": arch_type,
            "model_state" : self.model.state_dict(),
            "optimizer_state" : self.optimizer.state_dict()}
            , path)
        print(f"Model saved to {path}")
        
class Inference:
    def __init__(self, model, flatten = True):
        self.model = model
        self.flatten = flatten
    
    def load(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
    
    def predict(self, X):
        with torch.no_grad():
            if self.flatten:
                X = X.view(X.size(0), -1)
            output = self.model(X)
            return torch.argmax(output, dim=1)