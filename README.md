<div align="center">

# Tensorflow-LSTM
This is a simple implementation of LSTM model by using tensorFlow for classifying MNIST dataset.

For school, class BADE03.

</div>

---

## About


This is a simple implementation of LSTM model by using tensorFlow for classifying MNIST dataset.

Key features:
- **Dataset**: MNIST, processed with TensorFlow Datasets.
- **Evaluation Metric**: F1 Score (baseline target ≥ 98%).
- **Reproducibility**: Includes pre-trained weights and saved model.

---

## Result

| Metric    | F1 Score | Loss   |
|-----------|----------|--------|
| Result 1  | 0.9818   | 0.0760 |
| Result 2  | 0.9817   | 0.0732 |

---

## Requirements

- Python 3.7+
- tensorflow==2.18.0
- tensorflow-datasets==4.9.7
- numpy
- matplotlib

---

## Installation

## Automatic Installation
```bash
git clone https://github.com/PhotographyFlow/Tensorflow-LSTM.git
cd Tensorflow-LSTM
./Installation.bat
Done!
```
---

## Manual Installation
If the script doesn't work, set up manually:
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
.\venv\Scripts\activate        # Windows
pip install tensorflow==2.18.0 tensorflow-datasets==4.9.7
pip install numpy matplotlib
```
---

## Usage
Train and test the model.

### Train the model

Run this command to train the model:

```bash
python main.py
```
This will save:

Model weights to ./weights.weights.h5

Full model to ./LSTM_model.keras

### Test model

Run this command to test the trained model:

```bash
python test.py
```
Make sure the model file (./LSTM_model.keras) exists in the folder.

---

## References

<https://github.com/kaze-desu/LSTM-baseline-BADE03>

<https://tensorflow.google.cn>

<https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras>

Work in progress.

---

## Acknowledgements

Special thanks to Davison Wang for guiding this project in BADE03.

---

## Group Members

YingJi Zhao

Meiyi Qian 
