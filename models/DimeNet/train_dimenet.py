import os
import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models.dimenet import DimeNetPlusPlus
from torch_geometric.data import Dataset

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
    
# %%
class QM9AnalysisPipeline:
    # Mapping from property names to their fixed literature indices
    PROPERTY_TO_TARGET_INDEX = {
        "Dipole Moment": 0,
        "Isotropic polarizability": 1,
        "HOMO": 2,
        "LUMO": 3,
        "HOMO-LUMO gap": 4,
        "Electronic spatial extent": 5,
        "Zero point vibrational energy": 6,
        "Internal energy at 0K": 7,
        "Internal energy at 298.15K": 8,
    }

# List of all target property names
target_names = list(QM9AnalysisPipeline.PROPERTY_TO_TARGET_INDEX.keys())

# Corresponding list of all target indices
target_indices = list(QM9AnalysisPipeline.PROPERTY_TO_TARGET_INDEX.values())
# %%  
def train():
    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)

    model_dir = 'dimenet_pp'

    # K-fold cross-validation settings
    k_folds = 3
    batch_size = 64
    learning_rate = 1e-3
    epochs = 35
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Loop over each target index
    for target_index, target_name in zip(target_indices, target_names):

        safe_name = target_name.replace(" ", "_").replace('-', '_')

        print(f'===== Predicting target index: {target_index} ({safe_name})  =====')

        # Load and shuffle the QM9 dataset
        dataset_path = f"./../agent_results/results_{safe_name}_CLIP.pt"
        dataset_list = torch.load(dataset_path)
        dataset = ListDataset(dataset_list)
        dataset = dataset.shuffle()


        # Cross-validation loop
        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            print(f'--- Fold {fold + 1}/{k_folds} ---')

            # Create a TensorBoard SummaryWriter for each fold
            log_dir = f'logs/{model_dir}/target_{target_index}_fold_{fold+1}'
            writer = SummaryWriter(log_dir=log_dir)

            # Directory to save model weights, losses, etc.
            save_dir = f'runs/{model_dir}/target_{target_index}/fold_{fold + 1}'
            os.makedirs(save_dir, exist_ok=True)

            # Split dataset into train/test
            train_idx = torch.tensor(train_idx, dtype=torch.long)
            test_idx = torch.tensor(test_idx, dtype=torch.long)

            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]

            # Further split training set into train/val
            val_split = 0.1
            n_val = int(len(train_dataset) * val_split)
            val_dataset = train_dataset[-n_val:]
            train_dataset = train_dataset[:-n_val]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize the DimeNet++ model
            model = DimeNetPlusPlus(
                hidden_channels=32,
                out_channels=1,
                out_emb_channels=32,
                int_emb_size=32,
                basis_emb_size=32,
                num_blocks=2,
                num_radial=2,
                num_spherical=3,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.L1Loss()

            best_val_loss = float('inf')

            # Lists to store epoch losses for optional file logging
            train_losses = []
            val_losses = []

            for epoch in range(1, epochs + 1):
                # ---- Training ----
                model.train()
                epoch_train_loss = 0.0

                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()

                    # Forward pass
                    target_pred = model(data.z, data.pos, data.batch)

                    # Calculate prediction loss
                    loss = criterion(target_pred.squeeze(), data.y[:, target_index])
                    loss.backward()
                    optimizer.step()

                    epoch_train_loss += loss.item()

                avg_train_loss = epoch_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # ---- Validation ----
                model.eval()
                epoch_val_loss = 0.0

                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device)
                        target_pred = model(data.z, data.pos, data.batch)
                        val_loss = criterion(target_pred.squeeze(), data.y[:, target_index])
                        epoch_val_loss += val_loss.item()

                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                # Log to TensorBoard
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Val', avg_val_loss, epoch)

                print(
                    f"[Epoch {epoch:03d}] "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

                # Save the best model if the val loss improves
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_fold_{fold + 1}.pt'))

            # ---- Save train/val losses to a text file ----
            losses_path = os.path.join(save_dir, 'train_val_losses.txt')
            with open(losses_path, 'w') as f:
                for e in range(epochs):
                    f.write(f"Epoch {e+1}: Train Loss: {train_losses[e]:.4f}, "
                            f"Validation Loss: {val_losses[e]:.4f}\n")

            # ---- Test the best model ----
            model.load_state_dict(torch.load(os.path.join(save_dir, f'best_model_fold_{fold + 1}.pt')))
            model.eval()

            test_loss_sum = 0.0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    target_pred = model(data.z, data.pos, data.batch)
                    loss = criterion(target_pred.squeeze(), data.y[:, target_index])
                    test_loss_sum += loss.item()

            avg_test_loss = test_loss_sum / len(test_loader)
            print(f"[TEST] Fold {fold + 1} | Target Index {target_index} | Test MAE: {avg_test_loss:.4f}")

            # Log test loss to TensorBoard
            writer.add_scalar('Loss/Test', avg_test_loss, fold + 1)

            # Save test error to a text file
            test_errors_path = os.path.join(save_dir, 'test_errors.txt')
            os.makedirs(os.path.dirname(test_errors_path), exist_ok=True)
            with open(test_errors_path, 'a') as f:
                f.write(f"Fold {fold + 1} | Target Index {target_index} | Test MAE: {avg_test_loss:.4f}\n")

            # Close the writer at the end of this fold
            writer.close()


if __name__ == '__main__':
    train()
