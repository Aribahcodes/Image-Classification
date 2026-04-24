import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# custom linear layer with gates
# basically same as nn.Linear but with an extra gate parameter for each weight
class PrunableLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        # proper weight init
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_features))

        # gate scores init to -2.0 so sigmoid(-2) = 0.12
        # starting close to 0 means the l1 penalty can push them to 0 much faster
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features) * -2.0)

    def forward(self, x):
        # sigmoid squishes gate scores between 0 and 1
        gates = torch.sigmoid(self.gate_scores)

        # multiply weights by gates so pruned weights become ~0
        pruned_w = self.weight * gates

        # manual linear: x @ w.T + bias
        return x @ pruned_w.T + self.bias


class SelfPruningNet(nn.Module):

    def __init__(self):
        super().__init__()

        # 3072 because cifar10 images are 32x32x3
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        # flatten image first
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # no relu on last layer

        return x


# sparsity loss = sum of all gate values across all layers
def sparsity_loss(model):
    total = 0
    for layer in [model.fc1, model.fc2, model.fc3, model.fc4]:
        gates = torch.sigmoid(layer.gate_scores)
        total += gates.sum()
    return total


# calculate total number of weights once so we can normalize sparsity loss
def get_total_weights(model):
    total = 0
    for layer in [model.fc1, model.fc2, model.fc3, model.fc4]:
        total += layer.gate_scores.numel()
    return total


# function to check how many weights got pruned
def get_sparsity(model, threshold=0.01):
    total = 0
    pruned = 0
    for layer in [model.fc1, model.fc2, model.fc3, model.fc4]:
        gates = torch.sigmoid(layer.gate_scores)
        total += gates.numel()
        pruned += (gates < threshold).sum().item()
    return pruned / total


# load cifar10
transform = transforms.ToTensor()

train_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)


# training function so i can reuse it for different lambda values
def train_model(lam, epochs=20):
    print(f"\n--- training with lambda = {lam} ---")

    model = SelfPruningNet().to(device)

    # gates get a 10x higher learning rate so they actually move toward 0
    # weights and biases keep the normal lr
    optimizer = optim.Adam([
        {'params': [model.fc1.weight, model.fc2.weight, model.fc3.weight, model.fc4.weight,
                    model.fc1.bias,   model.fc2.bias,   model.fc3.bias,   model.fc4.bias],  'lr': 0.001},
        {'params': [model.fc1.gate_scores, model.fc2.gate_scores,
                    model.fc3.gate_scores, model.fc4.gate_scores], 'lr': 0.01}
    ])

    criterion = nn.CrossEntropyLoss()

    # calculate total weights once before training starts
    total_weights = get_total_weights(model)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            out = model(images)

            loss1 = criterion(out, labels)
            # normalize sparsity loss by total weights so it stays balanced
            loss2 = sparsity_loss(model) / total_weights
            loss = loss1 + lam * loss2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # check accuracy on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                out = model(images)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        sparsity = get_sparsity(model) * 100
        print(f"Epoch {epoch+1}/{epochs} | loss: {total_loss/len(train_loader):.4f} | acc: {acc:.2f}% | sparsity: {sparsity:.2f}%")

    return model, acc, sparsity


# trying 3 different lambda values like the question asked
# low = barely prunes, high = prunes aggressively but might hurt accuracy
lambdas = [0.1, 1.0, 5.0]

results = []  # will store (lam, acc, sparsity, model)

for lam in lambdas:
    model, acc, sparsity = train_model(lam, epochs=20)
    results.append((lam, acc, sparsity, model))


# print the results table
print("\n--- Results ---")
print(f"{'Lambda':<12} {'Test Accuracy':<16} {'Sparsity %':<12}")
print("-" * 40)
for lam, acc, sparsity, _ in results:
    print(f"{lam:<12} {acc:<16.2f} {sparsity:<12.2f}")


# plot gate distribution for all 3 models
# a good result should show a big spike near 0 (pruned weights)
# and a smaller cluster near 1 (weights the network kept)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (lam, acc, sparsity, model) in enumerate(results):

    # collect all gate values from all layers into one big list
    all_gates = []
    for layer in [model.fc1, model.fc2, model.fc3, model.fc4]:
        gates = torch.sigmoid(layer.gate_scores).detach().cpu().numpy().flatten()
        all_gates.extend(gates)

    axes[i].hist(all_gates, bins=50, color='steelblue', edgecolor='none')
    axes[i].set_title(f"lambda={lam}\nacc={acc:.1f}%  sparsity={sparsity:.1f}%")
    axes[i].set_xlabel("gate value")
    axes[i].set_ylabel("count")

    # red line to show the pruning threshold
    axes[i].axvline(x=0.01, color='red', linestyle='--', label='threshold 0.01')
    axes[i].legend()

plt.suptitle("Gate Value Distributions for Different Lambda Values")
plt.tight_layout()
plt.savefig("gate_distributions.png")
plt.show()
print("saved plot as gate_distributions.png")