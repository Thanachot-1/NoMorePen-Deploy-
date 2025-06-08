from flask import Flask, render_template, request, send_file, url_for
import torch
import os
import io
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image

app = Flask(__name__)

latent_dim = 100

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0),
            torch.nn.BatchNorm2d(256), torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128), torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 1, 4, 2, 1),
            torch.nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', images=None)

@app.route('/generate', methods=['POST'])
def generate():
    model_names = request.form['model_names']
    generated_images = []
    generator = Generator().to(device)
    for char in model_names:
        model_path = os.path.join(os.getcwd(), f"{char}.pth")
        if not os.path.exists(model_path):
            continue
        generator.load_state_dict(torch.load(model_path, map_location=device))
        z = torch.randn(1, latent_dim, 1, 1).to(device)
        with torch.no_grad():
            fake_img = generator(z).cpu()
        generated_images.append(fake_img)
    if not generated_images:
        return render_template('index.html', images=[])
    all_imgs = torch.cat(generated_images, dim=0)
    grid = make_grid(all_imgs, nrow=len(all_imgs), normalize=True, padding=0)
    ndarr = grid.mul(255).add(0.5).clamp(0,255).byte().permute(1,2,0).numpy()
    im = Image.fromarray(ndarr.squeeze())
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    buf.seek(0)
    img_url = url_for('image', img_id='result')
    # Save image to static folder for serving
    im.save(os.path.join(app.static_folder, 'result.png'))
    return render_template('index.html', images=[url_for('static', filename='result.png')])

@app.route('/image/<img_id>')
def image(img_id):
    img_path = os.path.join(app.static_folder, f"{img_id}.png")
    return send_file(img_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)