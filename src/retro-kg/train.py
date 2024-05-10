import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from data import TripleDataset
from tqdm import tqdm
from transE_MLP import TransE

DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 128
VAL_BATCH = 64
VAL_FREQ = 10
# N_TEMPLATES = 2990
N_TEMPLATES = 10225
LR = 0.00001
DROPOUT = 0.3
HIDDEN_SIZES = [512, 512, 512, 512, 512]
OUTPUT_DIM = 1024
MARGIN = 5.0
NORM = 2
ACTIVATION = nn.ReLU()
FP_DIM = 2048
N_EPOCHS = 5000
np.random.seed(42)


def evaluate(model, val_loader, n_templates):
    model.eval()
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    total = 0
    templates = torch.arange(n_templates).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        for heads, relations, tails in val_loader:
            current_batch_size = heads.size(0)
            heads = heads.to(torch.float).to(DEVICE)
            tails = tails.to(torch.float).to(DEVICE)
            relations = relations.to(DEVICE)

            all_templates = templates.repeat(current_batch_size, 1)
            heads = heads.repeat(n_templates, 1)
            tails = tails.repeat(n_templates, 1)

            scores = model.distance(heads, tails, all_templates.reshape(-1))
            scores = scores.view(current_batch_size, n_templates)

            # Calculate top-1 accuracy
            hits_1 += (scores.argmin(dim=1) == relations).sum().item()

            # Calculate top-3 accuracy
            _, top_3_indices = scores.topk(3, dim=1, largest=False)
            hits_3 += (top_3_indices == relations.unsqueeze(1)).any(dim=1).sum().item()

            # Calculate top-10 accuracy
            _, top_10_indices = scores.topk(10, dim=1, largest=False)
            hits_10 += (
                (top_10_indices == relations.unsqueeze(1)).any(dim=1).sum().item()
            )

            total += current_batch_size

    return hits_1 / total, hits_3 / total, hits_10 / total


def corrupt_batch(batch, molecule_fps):
    heads, relations, tails = batch
    head_or_relation = torch.randint(high=2, size=(heads.size(0),), device=DEVICE)
    random_heads = torch.randint(
        high=molecule_fps.shape[0], size=(heads.size(0),), device=DEVICE
    )
    random_relations = torch.randint(
        relations.max().item(), (heads.size(0),), device=DEVICE
    )
    # random_tails = torch.randint(
    #     high=molecule_fps.shape[0], size=(tails.size(0),), device=DEVICE
    # )

    corrupted_heads = torch.where(
        head_or_relation.view(-1, 1) == 0, molecule_fps[random_heads], heads
    )
    corrupted_relations = torch.where(
        head_or_relation == 1, random_relations, relations
    )
    # corrupted_tails = torch.where(
    #     head_or_relation.view(-1, 1) == 1, molecule_fps[random_tails], tails
    # )
    corrupted_triple = (corrupted_heads, corrupted_relations, tails)
    return corrupted_triple


def train(model, train_loader, molecule_fps, val_loader):
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = [MARGIN]
    val_1 = []
    val_3 = []
    val_10 = []
    pds = []
    nds = []

    # Training loop
    for epoch in range(N_EPOCHS):
        model.train()
        current_train_losses = []
        for batch in tqdm(train_loader):
            heads = batch[0].to(torch.float).to(DEVICE)
            relations = batch[1].to(DEVICE)
            tails = batch[2].to(torch.float).to(DEVICE)
            corrupted_heads, corrupted_relations, corrupted_tails = corrupt_batch(
                (heads, relations, tails), molecule_fps
            )

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss
            loss, pd, nd = model(
                (heads, tails, relations),
                (corrupted_heads, corrupted_tails, corrupted_relations),
            )
            loss.mean().backward()
            current_train_losses.append(loss.mean().item())
            pds.append(pd.mean().item())
            nds.append(nd.mean().item())
            optimizer.step()
        train_losses.append(np.mean(current_train_losses))

        # Evaluate model on validation set
        if (epoch) % VAL_FREQ == 0:
            hits_1, hits_3, hits_10 = evaluate(model, val_loader, N_TEMPLATES)
            val_1.append(hits_1 * 100)
            val_3.append(hits_3 * 100)
            val_10.append(hits_10 * 100)

            # Save matplotlib plot of training loss and top-10 accuracy

            plt.plot(train_losses)
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title("Training Loss")
            plt.savefig("output_test/train_loss.png")
            plt.close()

            plt.plot(val_1, label="Hits@1")
            plt.plot(val_3, label="Hits@3")
            plt.plot(val_10, label="Hits@10")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Validation Accuracy")
            plt.legend()
            plt.savefig("output_test/val_accuracy.png")
            plt.close()

        print(
            f"Epoch {epoch}, Training Loss: {train_losses[-1]},"
            f"PD: {np.mean(pds)}, ND: {np.mean(nds)},"
            f"Hits@1: {val_1[-1]}, Hits@3: {val_3[-1]}, Hits@10: {val_10[-1]}"
        )
        torch.save(model.state_dict(), "output_test/model.pth")

    return train_losses, hits_1, hits_3, hits_10


if __name__ == "__main__":
    # Load training data
    print("Loading training data...")
    train_heads = np.load("data/train_head_fps.npy")
    train_relations = np.load("data/train_relation_ids.npy")
    train_tails = np.load("data/train_tail_fps.npy")
    molecule_fps = torch.tensor(np.load("data/molecule_fps.npy"), device=DEVICE)

    # Train Dataset
    train_dataset = TripleDataset(train_heads, train_relations, train_tails)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    print(f"Loaded training data of size {len(train_dataset)}")

    # Load validation data
    print("Loading validation data...")
    val_heads = np.load("data/val_head_fps.npy")
    val_relations = np.load("data/val_relation_ids.npy")
    val_tails = np.load("data/val_tail_fps.npy")

    # Validation Dataset
    val_dataset = TripleDataset(val_heads, val_relations, val_tails)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=VAL_BATCH, shuffle=False
    )
    print(f"Loaded validation data of size {len(val_dataset)}")

    # Initialize model
    model = TransE(
        n_templates=N_TEMPLATES,
        device=DEVICE,
        norm=NORM,
        fp_dim=FP_DIM,
        dropout=DROPOUT,
        hidden_sizes=HIDDEN_SIZES,
        hidden_activation=ACTIVATION,
        output_dim=OUTPUT_DIM,
        margin=MARGIN,
    )

    # Move model and data to device
    model.to(torch.device(DEVICE))

    # Train model
    train(model, train_loader, molecule_fps, val_loader)
