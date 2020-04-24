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
