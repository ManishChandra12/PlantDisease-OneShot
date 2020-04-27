# PlantDisease-OneShot

## Project Setup
1. Clone the repo
2. Install pipenv
```
pip install pipenv
```
3. cd to the project directory
4. Create the virtual environment
```
pipenv install --skip-lock
```
5. Activate the virtual environment
```
pipenv shell
```

## Download the PlantDoc Dataset
```git clone https://github.com/pratikkayal/PlantDoc-Dataset.git```

## Generate cropped images
```python3 -m src.generate_cropped```

## Finetune RESNET on uncropped images
```python3 -m src.resnet```

## Finetune RESNET on cropped images
```python3 -m src.resnet --cropped```

## Generate training and validation data for Siamese network (uncropped images) (This may take a long time)
```python3 -m src.generate_siamese_datacsv```

## Generate training and validation data for Siamese network (cropped images) (This may take a long time)
```python3 -m src.generate_siamese_datacsv --cropped```

## Train Siamese network on uncropped images
```python3 -m src.siamese```

## Train Siamese network on cropped images
```python3 -m src.siamese --cropped```

## Extract image embeddings from the shared part of the Siamese network (uncropped images)
```python3 -m src.extract_embedding```

## Extract image embeddings from the shared part of the Siamese network (cropped images)
```python3 -m src.extract_embedding --cropped```
