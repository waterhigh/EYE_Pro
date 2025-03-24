import torch

from dataset import AgeDataset  # Assuming your dataset.py defines AgeDataset
from model import AgePredictor  # Assuming your model.py defines AgePredictor

# Load the model
model = AgePredictor()
model.load_state_dict(torch.load("model.pth"))  # Replace "model.pth" with your model's path
model.eval()

# Load the test dataset
test_dataset = AgeDataset(split="test")  # Assuming your dataset has a split argument
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Testing loop
def test_model():
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test_model()