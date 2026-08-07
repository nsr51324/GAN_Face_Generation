import torch
import torch.nn as nn
import streamlit as st
import numpy as np
from pathlib import Path
from torchvision.utils import make_grid
from PIL import Image

latent_dim = 100
features_g = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


@st.cache_resource(show_spinner=False)
def load_generator(model_path):
    model = Generator().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def tensor_to_image(tensor_batch, nrow):
    grid = make_grid(tensor_batch, nrow=nrow, padding=2, normalize=True)
    array = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


st.set_page_config(page_title="Face Generator", page_icon="🧑", layout="wide")

st.title("Human Face Generator with GAN")
st.markdown("This app uses a trained generator to create new human face images from random noise.")
st.markdown("The model is loaded from the project folder by default.")

base_dir = Path(__file__).resolve().parent
default_model_path = base_dir / "models" / "generator_final.pth"

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model file path", value=str(default_model_path))
    num_images = st.slider("Number of images", min_value=1, max_value=16, value=8, step=1)
    use_seed = st.checkbox("Use fixed seed", value=False)
    seed_value = st.number_input("Seed value", value=42, step=1, disabled=not use_seed)
    generate_clicked = st.button("Generate faces", type="primary", use_container_width=True)

model_path_obj = Path(model_path).expanduser()

if not model_path_obj.exists():
    st.warning(f"Model file not found at: {model_path_obj}")
    st.info("Please update the path or make sure the file exists in the models folder.")
    st.stop()

if generate_clicked:
    with st.spinner("Generating faces..."):
        generator = load_generator(str(model_path_obj))
        if use_seed:
            torch.manual_seed(int(seed_value))
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        with torch.no_grad():
            fake_images = generator(noise).detach().cpu()
        nrow = int(np.ceil(np.sqrt(num_images)))
        result_image = tensor_to_image(fake_images, nrow)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(result_image, caption=f"{num_images} generated face(s)")
    with col2:
        st.metric("Model file", model_path_obj.name)
        st.metric("Device", str(device))
        st.caption("Generated images come from random latent vectors.")
else:
    st.info("Click the button in the sidebar to start generating faces.")
