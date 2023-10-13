from datasets import load_dataset
import pandas as pd
import json

class FakeNewsDataset(): 

    def getData():    
        dataset = load_dataset('liar')
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        validation_dataset = dataset['validation']

        data = {"label": train_dataset['label'], "statement": train_dataset['statement'], "subject": train_dataset['subject']}
        pd_train_dataset = pd.DataFrame(data)
        data = {"label": test_dataset['label'], "statement": test_dataset['statement'], "subject": test_dataset['subject']}
        pd_test_dataset = pd.DataFrame(data)
        data = {"label": validation_dataset['label'], "statement": validation_dataset['statement'], "subject": validation_dataset['subject']}
        pd_validation_dataset = pd.DataFrame(data)

        train_analysis = []
        with open("training_analysis.txt", "r") as f:
            lines = [line.rstrip() for line in f]
            train_analysis = [line for line in lines if line]
        list_of_dicts = [json.loads(line) for line in train_analysis]
        analysis_df = pd.DataFrame(list_of_dicts)
        pd_train_dataset = pd.concat([pd_train_dataset, analysis_df], axis=1)

        test_analysis = []
        with open("test_analysis.txt", "r") as f:
            lines = [line.rstrip() for line in f]
            test_analysis = [line for line in lines if line]
        list_of_dicts = [json.loads(line) for line in test_analysis]
        analysis_df = pd.DataFrame(list_of_dicts)
        pd_test_dataset = pd.concat([pd_test_dataset, analysis_df], axis=1)

        validation_analysis = []
        with open("validation_analysis.txt", "r") as f:
            lines = [line.rstrip() for line in f]
            validation_analysis = [line for line in lines if line]
        list_of_dicts = [json.loads(line) for line in validation_analysis]
        analysis_df = pd.DataFrame(list_of_dicts)
        pd_validation_dataset = pd.concat([pd_validation_dataset, analysis_df], axis=1)

        train = pd_train_dataset
        test = pd_test_dataset
        validation = pd_validation_dataset
        return train, test, validation
    
