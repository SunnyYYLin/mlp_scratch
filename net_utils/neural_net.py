import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import tqdm
from torch.utils.tensorboard import SummaryWriter

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, std=1e-4):
        super(TwoLayerNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, num_classes).to(self.device)
        nn.init.normal_(self.fc1.weight, std=std)
        nn.init.normal_(self.fc2.weight, std=std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def predict(self, X):
        X = X.to(self.device)
        return torch.argmax(self.forward(X), dim=1)
      
    def loss(self, X, y=None, reg=0.0):
        X = X.to(self.device)
        loss = None
        N = X.size(0)
        scores = self.forward(X)
        if y is None:
            return scores
        
        y = y.to(self.device)
        grads = {}
        
        loss = F.cross_entropy(scores, y)
        loss += reg * (torch.sum(self.fc1.weight ** 2) + torch.sum(self.fc2.weight ** 2))
        loss.backward()
        for name, param in self.named_parameters():
            grads[name] = param.grad
        
        return loss, grads
    
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=1000, batch_size=200, verbose=False, iters_per_epoch = 100, tqdm_verbose=False):
        # Convert numpy arrays to torch tensors
        X, y = X.to(self.device), y.to(self.device)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        train_dataset = TensorDataset(X, y)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        train_iter = iter(train_loader)
        
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, learning_rate_decay, verbose=verbose)
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        writer = SummaryWriter()

        for it in tqdm.tqdm(range(num_iters), disable=not tqdm_verbose):
            try:
                X_batch, y_batch = next(train_iter)
            except StopIteration:
                # train_iter = iter(train_loader)
                pass
            
            # Compute loss and gradients using the current minibatch
            optimizer.zero_grad()
            loss = self.loss(X_batch, y_batch, reg)
            optimizer.step()
            loss_history.append(loss[0].item())
            writer.add_scalar('Loss/train', loss[0].item(), it)
            if verbose and it % iters_per_epoch == 0:
                # print(f'Iteration {it}/{num_iters}, loss = {loss[0].item()}')
                train_iter = iter(train_loader)
                lr_scheduler.step()
                train_acc_history.append(self.check_accuracy(train_loader))
                val_acc_history.append(self.check_accuracy(val_loader))
                writer.add_scalar('Accuracy/train', train_acc_history[-1], it)
                writer.add_scalar('Accuracy/val', val_acc_history[-1], it)

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }
        
    def check_accuracy(self, loader):
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for x, y in loader:
                scores = self.forward(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
        return float(num_correct) / num_samples
        
if __name__ == "__main__":
    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5

    def init_toy_model():
        torch.random.manual_seed(0)
        return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

    def init_toy_data():
        torch.random.manual_seed(1)
        X = 10 * torch.randn(num_inputs, input_size)
        y = torch.tensor([0, 1, 2, 2, 1])
        return X, y

    net = init_toy_model()
    X, y = init_toy_data()
    stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=True)

    print('Final training loss: ', stats['loss_history'][-1])

    # plot the loss history
    plt.plot(stats['loss_history'])
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.show()