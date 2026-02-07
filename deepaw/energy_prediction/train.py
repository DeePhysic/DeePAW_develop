"""
Training Script for Energy Prediction

Train energy prediction head using pretrained DeePAW embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from deepaw.energy_prediction.models import EnergyHead, ScalarEnergyHead
from deepaw.energy_prediction.dataset import EnergyDataset, collate_fn
from deepaw.extract_embeddings import AtomicEmbeddingExtractor


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        embeddings = batch['embeddings'].to(device)
        target = batch['energy'].to(device)
        batch_idx = batch['batch_idx'].to(device)

        optimizer.zero_grad()
        pred = model(embeddings, batch_idx=batch_idx)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in dataloader:
        embeddings = batch['embeddings'].to(device)
        target = batch['energy'].to(device)
        batch_idx = batch['batch_idx'].to(device)

        pred = model(embeddings, batch_idx=batch_idx)
        loss = criterion(pred, target)

        total_loss += loss.item()
        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mae = (preds - targets).abs().mean().item()

    return total_loss / len(dataloader), mae


def main(
    db_path: str,
    model_type: str = 'scalar',
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = None
):
    """
    Main training function.

    Args:
        db_path: Path to ASE database
        model_type: 'scalar' or 'equivariant'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize extractor
    print("Loading embedding extractor...")
    extractor = AtomicEmbeddingExtractor(device=device)

    # Create dataset
    print(f"Loading dataset from: {db_path}")
    dataset = EnergyDataset(
        db_path=db_path,
        extractor=extractor,
        precompute_embeddings=True
    )

    # Split dataset
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    # Create model
    if model_type == 'scalar':
        model = ScalarEnergyHead().to(device)
    else:
        model = EnergyHead().to(device)
    print(f"Model: {model_type}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_energy_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")

    print("Training complete!")
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True, help='Database path')
    parser.add_argument('--model', type=str, default='scalar', choices=['scalar', 'equivariant'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    main(args.db, args.model, args.epochs, args.batch_size, args.lr)
