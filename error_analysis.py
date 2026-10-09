import torch
from torch.utils.data import DataLoader

from dataset import load_records, SpokenDigitDataset
from model import SpokenDigitCNN


DATA_DIR = "data/recordings"

# Load records
records = load_records(DATA_DIR)

# Same speaker-aware split used during training
speakers = sorted(set(r[2] for r in records))

train_speakers = speakers[:-2]
val_speaker = speakers[-2]
test_speaker = speakers[-1]

test_records = [
    r for r in records
    if r[2] == test_speaker
]

print("Test speaker:", test_speaker)
print("Test samples:", len(test_records))


# Test dataset
dataset = SpokenDigitDataset(
    test_records,
    augment=False
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)


# Load model
model = SpokenDigitCNN()

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# Find errors
errors = []

index = 0

with torch.no_grad():

    for x, y in loader:

        logits = model(x)
        predictions = logits.argmax(dim=1)

        for true_label, predicted_label in zip(
            y.tolist(),
            predictions.tolist()
        ):

            if true_label != predicted_label:

                filepath = test_records[index][0]

                errors.append(
                    (
                        true_label,
                        predicted_label,
                        filepath
                    )
                )

            index += 1


print("\nTotal errors:", len(errors))


print("\n========== 6 -> 8 ==========")

for true, pred, filepath in errors:

    if true == 6 and pred == 8:
        print(filepath)


print("\n========== 9 -> 5 ==========")

for true, pred, filepath in errors:

    if true == 9 and pred == 5:
        print(filepath)


print("\n========== 0 -> 9 ==========")

for true, pred, filepath in errors:

    if true == 0 and pred == 9:
        print(filepath)