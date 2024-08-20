import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

# Define the model and move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) # decide which model

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

# Initialize variables for history
history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_accuracy': []
}

num_epochs = 10

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    epoch_time = time.time()

    for i, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'), 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate running loss
        running_loss += loss.data.item()

    # Normalize loss by the number of training batches
    running_loss /= len(train_loader)

    scheduler.step()

    # Calculate training and test accuracy
    train_accuracy = calculate_accuracy(model, train_loader, device)
    test_accuracy = calculate_accuracy(model, test_loader, device)

    # Save history
    history['train_loss'].append(running_loss)
    history['train_accuracy'].append(train_accuracy)
    history['test_accuracy'].append(test_accuracy)

    # Log information
    log = f"Epoch: {epoch} | Loss: {running_loss:.4f} | Training accuracy: {train_accuracy:.3f}% | Test accuracy: {test_accuracy:.3f}% | "
    epoch_time = time.time() - epoch_time
    log += f"Epoch Time: {epoch_time:.2f} secs"
    print(log)

print('==> Finished Training ...')
