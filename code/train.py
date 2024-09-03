import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from network import SafetyModel
from dataset import RobotSafetyDataset

from tqdm import tqdm

input_size = 74
hidden_size = 64
num_epochs = 20  # size of number of epoch
batch_size = 4  # size of each batch

torch.manual_seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Training at Device: {device}")

# If you are using a GPU, set the seed for CUDA as well
if device.type == 'cuda':
    torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42)  # if you are using multi-GPU


safety_net = SafetyModel(input_size=input_size, hidden_size=hidden_size).to(device=device)
criterion = nn.BCELoss()
optimizer = optim.Adam(safety_net.parameters(), lr=0.001)

# Instantiate the dataset
dataset = RobotSafetyDataset(root_dir=f'C:\\Users\\tanveer\\thesis\\safety-gymnasium-main\\SafetyCarGoalTrainDataset')

# Create a DataLoader for batching
# Split lengths: 80% for training, 10% for validation and rest of the percentange for test
# Define the split sizes
train_size = int(0.80 * len(dataset))
val_size = int(0.20 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for epoch in range(num_epochs):
    # Training Phase
    safety_net.train()  # Set the model to training mode
    running_train_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for batch_sequences, batch_label in train_loader:
            batch_sequences, batch_label = batch_sequences.to(device), batch_label.to(device)  # Move inputs and labels to device
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = safety_net(batch_sequences)
            loss = criterion(outputs, batch_label)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            pbar.set_postfix({'Train Loss': loss.item()})
            pbar.update(1)
    
    avg_train_loss = running_train_loss / len(train_loader)
    
    # Validation Phase
    safety_net.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_sequences, batch_label in val_loader:
            batch_sequences, batch_label = batch_sequences.to(device), batch_label.to(device)  # Move inputs and labels to device
            outputs = safety_net(batch_sequences)
            loss = criterion(outputs, batch_label)
            running_val_loss += loss.item()

            # Calculate accuracy (or other metrics) if needed
            predicted_labels = (outputs > 0.5).float()
            correct_predictions += (predicted_labels == batch_label).sum().item()
            total_predictions += batch_label.size(0) * batch_label.size(1)
    
    avg_val_loss = running_val_loss / len(val_loader)
    accuracy = (correct_predictions / total_predictions) * 100

    # Print or log the metrics
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, '
          f'Val Accuracy: {accuracy:.4f}')


# Save the trained safety network
torch.save(safety_net.state_dict(), 'safety_model.pth')

# Load the trained safety network
safety_net.load_state_dict(torch.load('safety_model.pth'))

# Set the model to evaluation mode
safety_net.eval()

# Initialize variables to track test loss and accuracy
running_test_loss = 0.0
correct_predictions = 0
total_predictions = 0

# Disable gradient calculation for testing
with torch.no_grad():
    for batch_sequences, batch_label in test_loader:
        batch_sequences, batch_label = batch_sequences.to(device), batch_label.to(device)
        
        # Forward pass
        outputs = safety_net(batch_sequences)
        loss = criterion(outputs, batch_label)
        
        # Accumulate test loss
        running_test_loss += loss.item()
        
        # Calculate accuracy (or other metrics)
        predicted_labels = (outputs > 0.5).float()
        correct_predictions += (predicted_labels == batch_label).sum().item()
        total_predictions += batch_label.size(0) * batch_label.size(1)

# Calculate average test loss and accuracy
avg_test_loss = running_test_loss / len(test_loader)
accuracy = (correct_predictions / total_predictions) * 100

# Print the test results
print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}')