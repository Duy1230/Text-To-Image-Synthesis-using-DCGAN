# Text-To-Image-Synthesis-using-DCGAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) for generating **Flower** images from textual descriptions. The model uses a pre-trained BERT model for text encoding and a custom generator to create images based on the encoded text.

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Duy1230/Text-To-Image-Synthesis-using-DCGAN.git
   cd Text-To-Image-Synthesis-using-DCGAN
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have a trained generator model saved as `checkpoints/generator.pth`.
2. Run the Streamlit interface:
   ```bash
   streamlit run interface.py
   ```
3. Open your web browser and navigate to `http://localhost:8501`.
4. Enter a text description in the input field and click "Generate Image" to see the generated image.
