import torch
from torch.utils.data import DataLoader

from dataset import load_records, SpokenDigitDataset
from model import SpokenDigitCNN


DATA_DIR = "data/recordings"

records = load_records(DATA_DIR)

speakers = sorted(set(r[2] for r in records))

print("Speakers:", speakers)


model = SpokenDigitCNN()

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


for speaker in speakers:

    speaker_records = [
        r for r in records
        if r[2] == speaker
    ]

    dataset = SpokenDigitDataset(
        speaker_records,
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    correct_6 = 0
    total_6 = 0

    correct_9 = 0
    total_9 = 0

    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for x, y in loader:

            logits = model(x)
            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == y
            ).sum().item()

            total_samples += len(y)

            for true, pred in zip(
                y.tolist(),
                predictions.tolist()
            ):

                if true == 6:
                    total_6 += 1
                    if pred == 6:
                        correct_6 += 1

                if true == 9:
                    total_9 += 1
                    if pred == 9:
                        correct_9 += 1

    print(
        f"\nSpeaker: {speaker}"
    )

    print(
        f"Overall accuracy: "
        f"{total_correct / total_samples:.2%}"
    )

    print(
        f"Digit 6 accuracy: "
        f"{correct_6}/{total_6} "
        f"({correct_6 / total_6:.2%})"
    )

    print(
        f"Digit 9 accuracy: "
        f"{correct_9}/{total_9} "
        f"({correct_9 / total_9:.2%})"
    )