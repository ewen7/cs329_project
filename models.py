from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch import nn
import torch.nn.functional as F

def init_model(args):
    """
    Initializes a model based on the value of args.model (either 'LR' or 'NN'?)
    """
    if args.dataset == 'spd':
        if args.model == 'lr':
            return LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Unknown model: {args.model}")
    elif args.dataset == 'hdp':
        if args.model == 'lr':
            return LogisticRegression(max_iter=1000)
        elif args.model == 'rfc':
            return RandomForestClassifier(criterion='entropy',n_estimators=20)
        else:
            raise ValueError(f"Unknown model: {args.model}")
    elif args.dataset == 'mnist':
        if args.model == 'cnn':
            return CNN()
        elif args.model == 'lr':
            return LogisticRegression(C=0.005, solver='saga', tol=0.1, penalty='l1')
        else:
            raise ValueError(f"Unknown model: {args.model}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
    def predict(self, x):
        y_hat = self.forward(x)
        return y_hat.argmax(dim=1).numpy()