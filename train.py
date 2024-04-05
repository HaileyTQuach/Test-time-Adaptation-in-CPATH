import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the ResNet50 model pre-trained on ImageNet
conv_base = models.resnet50(weights='DEFAULT')

# Since we're using include_top=False in Keras, we'll replace the FC layers ourselves
num_ftrs = conv_base.fc.in_features  # Get number of features in last layer
conv_base.fc = nn.Sequential(  # Replace the fully connected layer
    nn.Flatten(),
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Linear(256, 3),
    nn.Softmax(dim=1)
)

conv_base = conv_base.to(device)  # Move model to GPU if available

# Freeze all layers except 'layer4[0].conv1'
# Make the convolutional base trainable and unfreeze layers after 'layer4[0].conv1'
for name, child in conv_base.named_children():
    if name in ['layer4']:
        for param in child.parameters():
            param.requires_grad = True
    else:
        for param in child.parameters():
            param.requires_grad = False

# Data augmentation and normalization for training
# Just normalization for validation
train_transforms = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Assuming '/content/drive/MyDrive/COMP499/Project/Training data/02_training_native' is your training directory
train_dataset = ImageFolder(root='Training_data/02_training_native', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(conv_base.parameters(), lr=1e-5)

# Training loop
num_epochs = 7
for epoch in range(num_epochs):
    conv_base.train()  # Set the model to training mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')

    for batch_idx, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the configured device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = conv_base(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_preds / total_preds

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
torch.save(conv_base, 'Models/TvN_350_SN_D256_Initial_Ep7_fullmodel.pth')

# Clear PyTorch's CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
    
##Consecutive stepwise release of further convolutional layers
# v2, Layers 3, 4 free
conv_base.train()

for name, child in conv_base.named_children():
    if name == 'layer3':
        for param in child.parameters():
            param.requires_grad = True

# Update the optimizer to include the newly unfrozen parameters
optimizer = Adam(filter(lambda p: p.requires_grad, conv_base.parameters()), lr=1e-6)

# Continue training or fine-tuning
for epoch in range(1):  # Fine-tune for 1 epoch
    conv_base.train()  # Ensure the model is in training mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')

    for batch_idx, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the configured devic
        optimizer.zero_grad()

        outputs = conv_base(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    # Compute loss and accuracy
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_preds / total_preds
    print(f'Fine-tune Epoch, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

torch.save(conv_base, 'Models/TvN_350_SN_D256_v2_Ep1_fullmodel.pth')