import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

class FeatureCNN(nn.Module):
    def __init__(
        self,
        num_filters=32,
        hidden_dim=128,
        kernel_size=4,
        max_pool=2,
        output_dim=10,
        width=28,
        height=28
    ):
        super().__init__()

        # Feature extractor φ(x)
        self.feature_extractor = nn.Sequential(
            # Convolution 1
            nn.Conv2d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size),
            nn.ReLU(),

            # Convolution 2
            nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size = kernel_size),
            nn.ReLU(),

            # Max pooling
            nn.MaxPool2d(kernel_size=max_pool),

            nn.Dropout(p=0.25),
            # Flatten and fully connected (account for two convolutions and maxpooling in input dimension)
            nn.Flatten(),
            nn.Linear(in_features = (num_filters 
                                    * ((width - 2 * kernel_size + 2) // max_pool)
                                    * ((height - 2 * kernel_size + 2) // max_pool)), 
                        out_features = hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = 0.5)
        )

        # Deterministic classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        phi = self.feature_extractor(x)
        logits = self.classifier(phi)
        return phi, logits

# Feature CNN trained as a classifier
def train_feature_CNN(
    model,
    train_dataset,
    device,
    train_indices = None,
    batch_size=128,
    lr=3e-4,
    wd = 1e-4,
    n_epochs=50):

    if train_indices is None:
        train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
    else:
        train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=batch_size,
            shuffle=True
        )

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            _, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()


        # print(f"Epoch {epoch} loss = {epoch_loss}")
        # epoch_loss += loss.item() * y.size(0)
        # preds = logits.argmax(dim=1)
        # correct += (preds == y).sum().item()
        # total += y.size(0)

        # avg_loss = epoch_loss / total
        # acc = correct / total

        # print(
        #     f"Epoch {epoch+1:02d} | "
        #     f"loss = {avg_loss:.4f} | "
        #     f"acc = {acc:.4f}"

    return model
