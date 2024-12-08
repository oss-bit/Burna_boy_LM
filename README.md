# Burna Boy LM

**Burna Boy LM** is a Transformer-based language model designed to generate lyrics in the style of Burna Boy. It can produce creative and coherent Afrobeat-inspired lyrics given an initial phrase or sentence. This project is built with PyTorch and showcases the power of deep learning in creative applications.

---

## Features

- Generate Burna Boy-inspired lyrics from a starting phrase.
- Train the model on custom lyric datasets to fine-tune or adapt it to other artists.
- Supports easy inference and training with command-line arguments.
- Modular design for extensibility.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- PyTorch 1.12+ (with GPU support recommended)
- Other dependencies listed in `requirements.txt` (install using `pip install -r requirements.txt`).


### Installation

Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Burna_boy_LM.git

   cd Burna_boy_LM
   ```
Install the required dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Infrencing
To generate lyrics based on a starting phrase.
```bash
python3 transformersBLLM.py test --phrase "Yo what's up
```
### Training
To train the model with different hyperparameter settings.
```bash
python3 transformersBLLM.py train -h
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

---
Start generating Afrobeat magic today with Burna Boy LM! 🎶
