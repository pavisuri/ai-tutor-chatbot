# data_loader.py

from datasets import load_dataset

def load_tutorial_data():
    """
    Load the AI_Tutor dataset from Hugging Face using the datasets library.
    """
    dataset = load_dataset('Pavithrars/AI_dataset')
    return dataset['train']  # Return the training portion of the dataset
