# inference.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import CodespaceWeatherDataset
from model import ExtremeWeatherModel
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. Load the Test Data
    test_dataset = CodespaceWeatherDataset('testing/telangana_weather_test_data.nc', seq_length=24)
    # Batch size 1 makes it easier to track individual predictions
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 

    # 2. Initialize the Model and Load Weights
    model = ExtremeWeatherModel(input_channels=5, hidden_dim=16).to(device)
    model.load_state_dict(torch.load("convlstm_extreme_weather.pt", map_location=device))
    model.eval() # Set model to evaluation mode (turns off dropout/batchnorm updates)

    threshold = 0.1164 # Your 99th percentile threshold

    all_targets = []
    all_predictions = []

    print("Evaluating predictions...")
    with torch.no_grad(): # Turn off gradients to save memory and speed up inference
        for X, Y in test_loader:
            X = X.to(device)
            target_precip = Y[:, 0:1, :, :].to(device)
            
            # True Labels
            actual_extreme = (target_precip > threshold).float().cpu().numpy().flatten()
            
            # Model Predictions
            raw_logits = model(X)
            # Apply sigmoid because we didn't include it in the model's forward pass
            probabilities = torch.sigmoid(raw_logits) 
            # Convert probabilities > 0.5 into binary 1s and 0s
            predicted_extreme = (probabilities > 0.5).float().cpu().numpy().flatten()
            
            all_targets.extend(actual_extreme)
            all_predictions.extend(predicted_extreme)

    # 3. Calculate Metrics
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== INFERENCE RESULTS (September 2023 Unseen Data) ===")
    print(f"Precision: {precision:.4f} (When it predicts extreme rain, it is right {precision*100:.1f}% of the time)")
    print(f"Recall:    {recall:.4f} (It caught {recall*100:.1f}% of all actual extreme rain events)")
    print(f"F1-Score:  {f1:.4f} (The balance between Precision and Recall)")
    print("\nConfusion Matrix:")
    print(f"True Negatives (Correctly predicted no extreme rain): {cm[0][0]}")
    print(f"False Positives (False Alarms): {cm[0][1]}")
    print(f"False Negatives (Missed extreme events): {cm[1][0]}")
    print(f"True Positives (Successfully caught extreme rain): {cm[1][1]}")

if __name__ == "__main__":
    evaluate_model()