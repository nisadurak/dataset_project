from pathlib import Path
import random
import shutil

# ================== AYARLAR ==================

RAW_DIR = Path("dataset_raw")   # kaynak
OUT_DIR = Path("dataset")       # hedef
TRAIN_RATIO = 0.8               # %80 train, %20 val

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_images(category_dir: Path):

    imgs = []
    for p in category_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    return imgs


def main():
    if not RAW_DIR.exists():
        print(f"[HATA] RAW_DIR yok: {RAW_DIR.resolve()}")
        return


    (OUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "val").mkdir(parents=True, exist_ok=True)

    print(f"[+] Kaynak: {RAW_DIR.resolve()}")
    print(f"[+] Hedef : {OUT_DIR.resolve()}\n")

    category_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not category_dirs:
        print("[HATA] dataset_raw altında kategori klasörü yok.")
        return

    total_train = 0
    total_val = 0

    for cat_dir in sorted(category_dirs):
        cat_name = cat_dir.name  
        print(f"\n=== KATEGORİ: {cat_name} ===")

        images = collect_images(cat_dir)
        n = len(images)
        if n == 0:
            print("  [UYARI] Bu kategoride resim yok, atlanıyor.")
            continue

        print(f"  Toplam resim: {n}")

     
        random.shuffle(images)

        # Split index
        split_idx = int(n * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Hedef klasörler
        train_dir = OUT_DIR / "train" / cat_name
        val_dir = OUT_DIR / "val" / cat_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(train_imgs, start=1):
            dst = train_dir / src.name
            shutil.copy2(src, dst)

        for i, src in enumerate(val_imgs, start=1):
            dst = val_dir / src.name
            shutil.copy2(src, dst)

        print(f"  → train: {len(train_imgs)}")
        print(f"  → val  : {len(val_imgs)}")

        total_train += len(train_imgs)
        total_val += len(val_imgs)

    print("\n========== ÖZET ==========")
    print(f"Toplam train: {total_train}")
    print(f"Toplam val  : {total_val}")
    print("\n[✓] İşlem tamam. 'dataset/train' ve 'dataset/val'.")


if __name__ == "__main__":
    main()
