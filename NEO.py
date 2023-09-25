import numpy as np
from parser_1 import parameter_parser
from models.EncoderWeight import EncoderWeight
from models.EncoderAttention import EncoderAttention
from models.FNNModel import FNNModel
from preprocessing import get_graph_feature, get_pattern_feature

args = parameter_parser()

def main():
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature(vulnerability_type="reentrancy")
    pattern_train, pattern_test, label_by_extractor_train, label_by_extractor_valid = get_pattern_feature(vulnerability_type="reentrancy")

    graph_train = np.array(graph_train)
    graph_test = np.array(graph_test)

    # Extract pattern features using FNNModel
    fnn_model = FNNModel(pattern_train, pattern_test, label_by_extractor_train, label_by_extractor_valid)
    fnn_features_train = fnn_model.train()
    fnn_features_test = fnn_model.test()

    # Pass the FNN features through EncoderWeight for self-attention mechanism
    encoder_weight_model = EncoderWeight(graph_train, graph_test, fnn_features_train, fnn_features_test)
    self_attention_features_train = encoder_weight_model.train()
    self_attention_features_test = encoder_weight_model.test()

    # Pass the self-attention features through EncoderAttention for cross-attention mechanism
    encoder_attention_model = EncoderAttention(self_attention_features_train, self_attention_features_test, graph_experts_train, graph_experts_test)
    encoder_attention_model.train()
    encoder_attention_model.test()

if __name__ == "__main__":
    main()
