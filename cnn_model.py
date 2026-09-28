import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
os.makedirs("docs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
])

train_full = datasets.CIFAR10("data", train=True, download=True, transform=transform_train)
test_set = datasets.CIFAR10("data", train=False, download=True, transform=transform_test)

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
train_set, val_set = random_split(train_full, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,10)
        )
    def forward(self,x):
        return self.classifier(self.features(x))

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(epochs):
    model.train()
    correct = total = 0
    running_loss = 0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)
    train_losses.append(running_loss/len(train_loader))
    train_accs.append(correct/total)

    model.eval()
    val_loss = 0
    correct = total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            val_loss += criterion(out,y).item()
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)
    val_losses.append(val_loss/len(val_loader))
    val_accs.append(correct/total)
    print(f"Epoch {epoch+1}/{epochs} | train acc {train_accs[-1]:.4f} | val acc {val_accs[-1]:.4f}")

torch.save(model.state_dict(), "models/saved_cnn_model.pth")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.tight_layout(); plt.savefig("visualizations/training_loss.png"); plt.close()

plt.figure()
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.tight_layout(); plt.savefig("visualizations/training_accuracy.png"); plt.close()

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x,y in test_loader:
        out = model(x.to(device))
        y_true.extend(y.numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

report = classification_report(y_true, y_pred, target_names=classes)
print(report)
Path("docs/classification_report.txt").write_text(report)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(9,9))
disp.plot(ax=ax, xticks_rotation=45)
plt.tight_layout()
plt.savefig("visualizations/confusion_matrix.png")
plt.close()

Path("docs/evaluation_report.md").write_text(
    "# Evaluation Report\n\n"
    "The CNN was trained on CIFAR-10 with normalization, data augmentation, "
    "Batch Normalization, Dropout, and Adam optimization. "
    "The generated classification report and confusion matrix are included "
    "after running `python src/cnn_model.py`.\n"
)
print("Completed. Model and visualizations saved.")
