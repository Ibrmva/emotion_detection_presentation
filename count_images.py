from pathlib import Path

DATA_DIR = Path("data/fer2013")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif"
}

def count_images(folder):
    return sum(
        1 for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )

def show_dataset(folder):
    total = 0

    print(f"\n{folder.name.upper()} DATASET")
    print("-" * 40)

    for class_folder in sorted(folder.iterdir()):
        if class_folder.is_dir():
            count = count_images(class_folder)
            total += count
            print(f"{class_folder.name:15} {count:>7} images")

    print("-" * 40)
    print(f"TOTAL: {total} images")

    return total


train_total = show_dataset(DATA_DIR / "train")
test_total = show_dataset(DATA_DIR / "test")

print("\n" + "=" * 40)
print(f"TRAINING IMAGES: {train_total}")
print(f"TEST IMAGES:     {test_total}")
print(f"TOTAL IMAGES:    {train_total + test_total}")
print("=" * 40)