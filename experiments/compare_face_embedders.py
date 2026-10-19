"""Phase 1 decision support: does the ImageNet-pretrained ViT in
module1_face_recognition behave like a face embedding at all, compared
to a VGGFace2-pretrained InceptionResnetV1 (facenet-pytorch)?

A real face embedder should score a same-person pair clearly higher
than a different-person pair. This script measures that gap for both
models on the same photos so the comparison is a number, not a guess.

Usage:
    python experiments/compare_face_embedders.py same1.jpg same2.jpg other.jpg

    same1.jpg, same2.jpg : two different photos of the SAME person
    other.jpg            : a photo of a DIFFERENT person

Requires the packages already listed in requirements.txt (torch, timm,
facenet-pytorch). Run with your normal module1/module2 environment,
not the CPU-pivot requirements-cpu.txt one.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from module1_face_recognition.face_recognizer import FaceRecognizer

from facenet_pytorch import InceptionResnetV1, fixed_image_standardization


def vggface2_embedding(model, image):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: fixed_image_standardization(t * 255)),
    ])
    tensor = transform(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.squeeze(0).numpy()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    same1, same2, other = (Image.open(p) for p in sys.argv[1:4])

    print("Loading ImageNet-pretrained ViT (current module1 approach)...")
    vit = FaceRecognizer(device="cpu")

    print("Loading VGGFace2-pretrained InceptionResnetV1...")
    vgg = InceptionResnetV1(pretrained="vggface2").eval()

    vit_same = cosine(vit.extract_embedding(same1), vit.extract_embedding(same2))
    vit_diff = cosine(vit.extract_embedding(same1), vit.extract_embedding(other))
    vgg_same = cosine(vggface2_embedding(vgg, same1), vggface2_embedding(vgg, same2))
    vgg_diff = cosine(vggface2_embedding(vgg, same1), vggface2_embedding(vgg, other))

    print(f"\n{'Method':<16}{'Same-person':<14}{'Different-person':<18}{'Gap (higher = better)'}")
    print(f"{'ViT (ImageNet)':<16}{vit_same:<14.4f}{vit_diff:<18.4f}{vit_same - vit_diff:.4f}")
    print(f"{'VGGFace2':<16}{vgg_same:<14.4f}{vgg_diff:<18.4f}{vgg_same - vgg_diff:.4f}")
    print("\nA trustworthy face embedder should show a clear, consistent gap here.")
    print("A near-zero or inconsistent gap means that embedding space is not")
    print("actually separating identities, regardless of what threshold you pick.")


if __name__ == "__main__":
    main()
