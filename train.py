import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CodespaceWeatherDataset
from model import ExtremeWeatherModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Hyperparameters
    epochs = 20
    batch_size = 16 # Increase this on your droplet! 
    threshold = 0.1164 # Example threshold (You'll need to calculate this based on your normalized 'tp' distribution)

    print("Loading data...")
    dataset = CodespaceWeatherDataset('supporting_files/telangana_weather_data.nc', seq_length=24)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ExtremeWeatherModel(input_channels=5, hidden_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Calculate pos_weight for BCEWithLogitsLoss based on class imbalance
    # If 10% of pixels are extreme, pos_weight should be roughly 9.0
    pos_weight = torch.tensor([10.0]).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (X, Y) in enumerate(dataloader):
            X = X.to(device)
            
            # Y shape is (B, C, H, W). We only want Total Precipitation (index 0)
            target_precip = Y[:, 0:1, :, :].to(device)
            
            # Convert continuous rainfall into binary classes (1 for extreme, 0 for normal)
            target_class = (target_precip > threshold).float()

            optimizer.zero_grad()
            predictions = model(X)
            
            loss = criterion(predictions, target_class)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")

    # Save the model weights
    torch.save(model.state_dict(), "convlstm_extreme_weather.pt")
    print("Model saved successfully!")

if __name__ == "__main__":
    train()