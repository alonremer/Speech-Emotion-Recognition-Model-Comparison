import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Dataset class for CREMA-D
class CremaDataset(Dataset):
    def __init__(self, df, transform=None, target_size=(128, 256)):
        """
        Custom Dataset for the CREMA-D dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the paths and labels for training data.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (tuple): Target size to which the spectrogram will be padded or trimmed.
        """
        self.df = df
        self.transform = transform
        self.label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5}
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Fetches the spectrogram and label for the given index.

        Args:
            idx (int): Index of the data.

        Returns:
            torch.Tensor: Processed spectrogram.
            int: Corresponding label.
        """
        spectrogram = self.df.iloc[idx, 2]  # Assuming spectrogram is in the 3rd column
        label = self.df.iloc[idx, 0]
        label = self.label_map[label]

        # Resize spectrogram to target size (pad or trim)
        if spectrogram.size(0) < self.target_size[0]:
            # Pad to target size
            pad_size = (0, self.target_size[1] - spectrogram.size(1))
            spectrogram = F.pad(spectrogram, pad_size, mode='constant', value=0)
        elif spectrogram.size(0) > self.target_size[0]:
            # Trim to target size
            spectrogram = spectrogram[:self.target_size[0], :self.target_size[1]]
        else:
            # If width is shorter, pad width
            if spectrogram.size(1) < self.target_size[1]:
                pad_size = (0, self.target_size[1] - spectrogram.size(1))
                spectrogram = F.pad(spectrogram, pad_size, mode='constant', value=0)
            elif spectrogram.size(1) > self.target_size[1]:
                # Trim to target size
                spectrogram = spectrogram[:, :self.target_size[1]]

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

# Function to calculate the accuracy of the model
def calculate_accuracy(model, dataloader, device):
    """
    Calculates the accuracy of the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the data to evaluate.
        device (torch.device): Device on which to perform calculations.

    Returns:
        float: Accuracy of the model.
    """
    model.eval()  # Evaluation mode (disables dropout and batch normalization)
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    model_accuracy = total_correct / total_images * 100
    return model_accuracy
