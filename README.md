# Recurrent Attention Models for Classification of Low-Resolution Histopathology Data

Final project for CISC 881, applying a recurrent attention model (RAM) to whole-slide histopathology classification.

## Downloading the data

The dataset is publicly available on the [CAMELYON-16 grand challenge site](https://camelyon16.grand-challenge.org/), where the samples are able to be downloaded.


## Running the code

To download dependencies required to run this code, run

```bash
conda env create -f camelyon.yml
conda activate camelyon
```

If using the dataset downloaded from the CAMELYON-16 site, data preprocessing is required before running the model. To do this, run

```bash
python segment.py
```

This will take some time, but afterwards the model should be good to run. To run the model:

```bash
python main.py
```

## Changing Model Modes

If you're only interested in running the model in inference mode, edit the `params` dict in `main.py`. 
