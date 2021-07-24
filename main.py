from datasets import get_data
from models import NER


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arman_training = get_data("./data/arman/train.txt")
    arman_validation = get_data("./data/arman/dev.txt")
    arman_test = get_data("./data/arman/test.txt")

    # peyma_training = get_data("./data/peyma/train.txt")
    # peyma_validation = get_data("./data/peyma/dev.txt")
    # peyma_test = get_data("./data/peyma/test.txt")

    # hyperparameters for network
    dropout = 0.1
    # hyperparameters for training
    training_hyperparameters = {
        'epochs': 5,
        'warmup_steps': 500,
        'train_batch_size': 13,
        'learning_rate': 0.0001
    }

    # transformer = 'HooshvareLab/bert-base-parsbert-ner-uncased'
    transformer = 'HooshvareLab/bert-base-parsbert-peymaner-uncased'
    arman_tag_scheme = [
        'B-org',
        'I-org',
        'B-loc',
        'I-loc',
        'B-fac',
        'I-fac',
        'B-event',
        'I-event',
        'B-pro',
        'I-pro',
        'B-pers',
        'I-pers',
    ]

    # peyma_tag_scheme = [
    #     'B_ORG',
    #     'I_ORG',
    #     'B_LOC',
    #     'I-LOC',
    #     'B_DAT',
    #     'I_DAT',
    #     'B_MON',
    #     'I_MON',
    #     'B_TIM',
    #     'I_TIM',
    #     'B_PER',
    #     'I_PER',
    #     'B_PCT',
    #     'I_PCT'
    # ]

    arman_model = NER(
        dataset_training=arman_training,
        dataset_validation=arman_validation,
        tag_scheme=arman_tag_scheme,
        tag_outside='O',
        transformer=transformer,
        dropout=dropout,
        hyperparameters=training_hyperparameters
    )

    # peyma_model = NER(
    #     dataset_training=peyma_training,
    #     dataset_validation=peyma_validation,
    #     tag_scheme=peyma_tag_scheme,
    #     tag_outside='O',
    #     transformer=transformer,
    #     dropout=dropout,
    #     hyperparameters=training_hyperparameters
    # )

    arman_model.train()
    print(arman_model.evaluate_performance(arman_test))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
