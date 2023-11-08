from datasets import load_dataset
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from joblib import dump

from sklearn.preprocessing import StandardScaler


class FakeNewsDataset(): 

    def __init__(self) -> None:
        pass

    def get_common_columns_from_analysis(self, file_list):
        common_columns = None
        for file in file_list:
            with open(file, "r") as f:
                lines = [line.rstrip() for line in f if line.rstrip()]
                list_of_dicts = [json.loads(line) for line in lines]
                # If common_columns is not yet initialized, initialize it with columns from the first file
                if common_columns is None:
                    common_columns = set(list_of_dicts[0].keys())
                else:
                    # Find the intersection of current columns with previous columns
                    common_columns &= set(list_of_dicts[0].keys())
        return common_columns

    def append_analysis_to_dataset(self, dataset, analysis_file, common_columns):
        with open(analysis_file, "r") as f:
            lines = [line.rstrip() for line in f if line.rstrip()]
            list_of_dicts = [json.loads(line) for line in lines]
            list_of_common_dicts = [{key: d[key] for key in common_columns} for d in list_of_dicts]
            analysis_df = pd.DataFrame(list_of_common_dicts)
            return pd.concat([dataset, analysis_df], axis=1)


    def getData1(self):    
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



        common_columns = self.get_common_columns_from_analysis(["training_analysis.txt", "test_analysis.txt", "validation_analysis.txt"])

        pd_train_dataset = self.append_analysis_to_dataset(pd_train_dataset, "training_analysis.txt", common_columns)
        pd_test_dataset = self.append_analysis_to_dataset(pd_test_dataset, "test_analysis.txt", common_columns)
        pd_validation_dataset = self.append_analysis_to_dataset(pd_validation_dataset, "validation_analysis.txt", common_columns)

        mapping_function = lambda x: 0 if x in [0, 4, 5] else 1

        pd_train_dataset["label"] = pd_train_dataset["label"].map(mapping_function)
        pd_test_dataset["label"] = pd_test_dataset["label"].map(mapping_function)
        pd_validation_dataset["label"] = pd_validation_dataset["label"].map(mapping_function)
        combined_df = pd.concat([pd_train_dataset, pd_test_dataset, pd_validation_dataset], axis=0, ignore_index=True)

        label_counts = combined_df['label'].value_counts()

        # Separate majority and minority classes
        df_majority = combined_df[combined_df['label'] == 1]
        df_minority = combined_df[combined_df['label'] == 0]

        # Undersample the majority class
        df_majority_undersampled = df_majority.sample(len(df_minority), random_state=42)  # setting a random_state for reproducibility

        balanced_training_first, balanced_test_first = train_test_split(df_majority_undersampled, test_size=0.1, random_state=42)  # Setting a random state for reproducibility
        balanced_training_second, balanced_test_second = train_test_split(df_minority, test_size=0.1, random_state=42)  # Setting a random state for reproducibility

        balanced_training = pd.concat([balanced_training_first, balanced_training_second], axis=0)
        balanced_test = pd.concat([balanced_test_first, balanced_test_second], axis=0)


        # List of numerical columns to scale
        columns_to_scale = [column_name for column_name, dtype in balanced_training.dtypes.items() if dtype in ['float64', 'int64'] and column_name not in ['statement', 'subject', 'label']]

        # Initialize the standard scaler
        scaler = StandardScaler()

        # Fit the scaler on the data
        scaler.fit(balanced_training[columns_to_scale])
        dump(scaler, 'standard_scaler.pkl')

        # Transform the data
        balanced_training[columns_to_scale] = scaler.transform(balanced_training[columns_to_scale])
        balanced_test[columns_to_scale] = scaler.transform(balanced_test[columns_to_scale])

        train = balanced_training
        test = balanced_test
        validation = pd_validation_dataset
        return train, test, validation
    
    
   