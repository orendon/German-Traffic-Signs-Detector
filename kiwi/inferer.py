from kiwi.utils import CLASS_LABELS
from PIL import Image, ImageDraw

def add_label(file, label):
  img = Image.open(file)
  img = img.resize((200, 200))
  draw = ImageDraw.Draw(img)
  draw.line((0, 0) + (200,0), fill=255, width=20)
  draw.text((0, 0), label, fill=(0, 0, 0))

  return img

def display_inferences(model, X_data, files):
  print("\nPredictions summary:")
  predictions = model.predict(X_data)
  for i in range(len(predictions)):
    class_label = CLASS_LABELS[str(predictions[i])]
    print(class_label)

    img = add_label(files[i], class_label)
    img.show()
