# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from db_to_df import prepare_dataframe

# ------------------------------------
# 1. Create a Custom PyTorch Dataset
# ------------------------------------
class ASLDataset(Dataset):
    """
    A custom dataset that expects a Pandas DataFrame with columns:
      - 'label': the class label (string)
      - 'landmarks': a list of (x, y, z) tuples
    """
    def __init__(self, dataframe, label_to_idx):
        super().__init__()
        self.df = dataframe
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_str = row['label']
        label_idx = self.label_to_idx[label_str]

        # If landmarks is None, create a zero vector
        # (We treat 'nothing' as zeroed-out input.)
        if row['landmarks'] is None:
            # 21 landmarks * 3 coords = 63
            landmarks_np = np.zeros((63,), dtype=np.float32)
        else:
            # Flatten the 21 (x,y,z) landmarks into a single 63-length vector
            flattened = []
            for (x, y, z) in row['landmarks']:
                flattened.extend([x, y, z])
            landmarks_np = np.array(flattened, dtype=np.float32)

        return landmarks_np, label_idx

# ------------------------------
# 2. Define a Simple Model
# ------------------------------
class SimpleASLModel(nn.Module):
    """
    A simple feed-forward network for classification:
    Input: 63-dim (21 landmarks * 3 coords)
    Output: #classes (letters, plus 'nothing', etc.)
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # final layer (logits)
        return x

# --------------------------
# 3. Evaluate Model on Test
# --------------------------
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # outputs: [batch_size, num_classes]
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy

# --------------------------
# 4. Training Script
# --------------------------
def train_model(db_path='../landmarks.db', model_out='../asl_model.pth', epochs=50, batch_size=128):
    # 1) Load and split data
    df_train, df_test = prepare_dataframe(db_path)

    # 2) Build label dictionaries from the train set
    #    (or union of train+test if you prefer)
    all_labels = sorted(df_train['label'].unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    # Print label order
    print("Training Labels Order:", all_labels)

    num_classes = len(all_labels)

    # 3) Create Datasets and DataLoaders
    train_dataset = ASLDataset(df_train, label_to_idx)
    test_dataset  = ASLDataset(df_test,  label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 4) Define Model, Loss, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleASLModel(input_dim=63, hidden_dim=128, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5) Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)       # shape: [batch_size, num_classes]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate on test set at the end of each epoch
        accuracy = evaluate(model, test_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}")

    # 6) Save the Model + Label Mapping
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label
    }, model_out)

    print(f"Model trained and saved to {model_out}")

if __name__ == '__main__':
    train_model()
