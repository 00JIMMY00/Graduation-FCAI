import os
from imutils import paths
import face_recognition
import pickle
import cv2

print("[INFO] start processing faces...")
# Update the dataset path if necessary
imagePaths = list(paths.list_images("face_recognition/src/assets/dataset"))
knownEncodings = []
knownNames = []
print(imagePaths)

# Process each image
for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]
    
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# Specify the path for the encodings file inside assets
pickle_file_path = "face_recognition/src/assets/encodings.pickle"

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(pickle_file_path, "wb") as f:
    f.write(pickle.dumps(data))

print(f"[INFO] Training complete. Encodings saved to '{pickle_file_path}'")
