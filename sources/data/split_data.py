
from dataset import CodeDataset



def splits_data():

    dataset = CodeDataset(args=args,
                          dataset_name=name,
                          mode=mode,
                          task=task,
                          language=language,
                          split=split,
                          clone_mapping=clone_mapping,
                          stage=stage)