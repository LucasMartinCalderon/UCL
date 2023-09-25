import numpy as np


def get_pattern_feature(vulnerability_type):
    """
    Extract pattern features for the given vulnerability type from the corresponding files.
    """
    base_path = f"./graph_feature/{vulnerability_type}/"
    train_total_name_path = base_path + "contract_name_train.txt"
    test_total_name_path = base_path + "contract_name_valid.txt"
    pattern_feature_path = f"./pattern_feature/feature_FNN/{vulnerability_type}/"

    final_pattern_feature_train = []
    pattern_feature_train_label_path = f"./pattern_feature/feature_zeropadding/{vulnerability_type}/label_by_extractor_train.txt"

    final_pattern_feature_test = []
    pattern_feature_test_label_path = f"./pattern_feature/feature_zeropadding/{vulnerability_type}/label_by_extractor_valid.txt"

    # Extract pattern features for training data
    with open(train_total_name_path, 'r') as f_train:
        lines = f_train.readlines()
        for line in lines:
            line = line.strip('\n').split('.')[0]
            tmp_feature = np.loadtxt(pattern_feature_path + line + '.txt')
            final_pattern_feature_train.append(tmp_feature)

    # Extract pattern features for validation data
    with open(test_total_name_path, 'r') as f_test:
        lines = f_test.readlines()
        for line in lines:
            line = line.strip('\n').split('.')[0]
            tmp_feature = np.loadtxt(pattern_feature_path + line + '.txt')
            final_pattern_feature_test.append(tmp_feature)

    # Extract labels produced by the extractor (training data)
    label_by_extractor_train = []
    with open(pattern_feature_train_label_path, 'r') as f_train_label_extractor:
        labels = f_train_label_extractor.readlines()
        for label in labels:
            label_by_extractor_train.append(label.strip('\n'))

    # Extract labels produced by the extractor (validation data)
    label_by_extractor_valid = []
    with open(pattern_feature_test_label_path, 'r') as f_test_label_extractor:
        labels = f_test_label_extractor.readlines()
        for label in labels:
            label_by_extractor_valid.append(label.strip('\n'))

    return final_pattern_feature_train, final_pattern_feature_test, label_by_extractor_train, label_by_extractor_valid


def get_graph_feature(vulnerability_type):
    """
    Extract graph features for the given vulnerability type from the corresponding files.
    """
    base_path = f"./graph_feature/{vulnerability_type}/"
    graph_feature_train_data_path = base_path + f"{vulnerability_type}_final_train.txt"
    graph_feature_train_label_path = base_path + "label_by_experts_train.txt"

    graph_feature_test_data_path = base_path + f"{vulnerability_type}_final_valid.txt"
    graph_feature_test_label_path = base_path + "label_by_experts_valid.txt"

    # Extract labels by experts for training data
    label_by_experts_train = []
    with open(graph_feature_train_label_path, 'r') as f_train_label_expert:
        labels = f_train_label_expert.readlines()
        for label in labels:
            label_by_experts_train.append(label.strip('\n'))

    # Extract labels by experts for validation data
    label_by_experts_valid = []
    with open(graph_feature_test_label_path, 'r') as f_test_label_expert:
        labels = f_test_label_expert.readlines()
        for label in labels:
            label_by_experts_valid.append(label.strip('\n'))

    # Extract graph features for training data
    graph_feature_train = np.loadtxt(graph_feature_train_data_path).tolist()
    for i in range(len(graph_feature_train)):
        graph_feature_train[i] = [graph_feature_train[i]]

    # Extract graph features for validation data
    graph_feature_test = np.loadtxt(graph_feature_test_data_path, delimiter=",").tolist()
    for i in range(len(graph_feature_test)):
        graph_feature_test[i] = [graph_feature_test[i]]

    return graph_feature_train, graph_feature_test, label_by_experts_train, label_by_experts_valid


if __name__ == "__main__":
    # List of vulnerability types to be processed
    vulnerabilities = ["reentrancy", "timestamp", "unsafe_delegatecall"]

    # Iterate over each vulnerability type and extract pattern and graph features
    for vulnerability in vulnerabilities:
        pattern_train, pattern_test, pattern_experts_train, pattern_experts_test = get_pattern_feature(vulnerability)
        graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature(vulnerability)
        print(f"Processed {vulnerability}")
