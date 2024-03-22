from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms
from cnn_training import CNet
from qnn_training import HNet, create_qnn

app = FastAPI()

# Set the conversion of images for upload
transform = transforms.Compose([
    lambda image: image.convert('RGB'),  # Convert images to RGB
    transforms.Resize((32, 32)),  # Resize the image
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Load weights for classic CNN models and hybrid QNN models
model_cnn = CNet()
model_cnn.load_state_dict(torch.load("model_cnet.pt"))
model_cnn.eval()

qnn = create_qnn()
model_qnn = HNet(qnn)
model_qnn.load_state_dict(torch.load("model_hybrid.pt"))
model_qnn.eval()


# Define routing for classic CNN model prediction
@app.post("/predict/cnn/")
async def predict_cnn(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Make sure the image is RGB, and then apply the conversion
    image = transform(image)

    with torch.no_grad():
        prediction = model_cnn(image.unsqueeze(0))  # Increase the batch dimension for prediction
    predicted_class = torch.argmax(prediction, dim=1).item()
    return JSONResponse(content={"prediction": predicted_class})


@app.post("/predict/qnn/")
async def predict_qnn(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Apply preprocessing conversion
    image = transform(image)

    with torch.no_grad():
        prediction = model_qnn(image.unsqueeze(0))  # Increase the batch dimension for prediction
    predicted_class = torch.argmax(prediction, dim=1).item()
    return JSONResponse(content={"prediction": predicted_class})


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI model serving."}
