from unstructured.partition.pdf import partition_pdf
from PIL import Image
import pytesseract

def load_file(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return [f.read()]

    if path.endswith(".pdf"):
        elements = partition_pdf(path)
        return [el.text for el in elements if el.text]

    if path.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return [text]

    raise ValueError("Unsupported file type")
