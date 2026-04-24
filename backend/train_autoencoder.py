import cv2
import torch
import numpy as np
from model import VideoAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_path = "../sample_videos/normal.mp4"

model = VideoAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

cap = cv2.VideoCapture(video_path)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    frames.append(gray)

cap.release()

frames = np.array(frames) / 255.0
frames = torch.tensor(frames).unsqueeze(1).float().to(device)

print(f"Training frames: {frames.shape}")

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(frames)
    loss = criterion(outputs, frames)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/10, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "models/autoencoder.pth")

print("Model saved to models/autoencoder.pth")