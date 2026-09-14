from ALSTM.ALSTM import ALSTMModel
from GAT.GAT import GATM
from GRU.GRU import GRUModel
from Localformer.Localformer import Transformer as Localformer
from TCN.TCN import TCNModel
from Transformer.Transformer import Transformer

# generates for every model a parameter configuration suggestion and initializes the model with this configuration
DROPOUT = 0.2 # is currently for every model hard coded

# when called initializes one parameter configuration suggested by optuna
def get_alstm_builder(trial, d_feat):
    hidden_size = trial.suggest_categorical("alstm_hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("alstm_num_layers", 3, 6)

    # 2. builder function that is returned
    def create_model():
        return ALSTMModel(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=DROPOUT
        )

    # create dummy model to calculate number of params
    dummy_model = create_model()
    # calculates the number of trainable params and concatenates the config into a string
    n_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    config_str = f"hidden_size:{hidden_size} | num_layers:{num_layers} | params:{n_params}"
    return create_model, config_str, n_params


def get_gat_builder(trial, d_feat):
    hidden_size = trial.suggest_categorical("gat_hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("gat_num_layers", 3, 6)

    # 2. builder function that is returned
    def create_model():
        return GATM(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=DROPOUT
        )

    # create dummy model to calculate number of params
    dummy_model = create_model()
    # calculates the number of trainable params and concatenates the config into a string
    n_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    config_str = f"hidden_size:{hidden_size} | num_layers:{num_layers} | params:{n_params}"
    return create_model, config_str, n_params


def get_gru_builder(trial, d_feat):
    hidden_size = trial.suggest_categorical("gru_hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("gru_num_layers", 3, 6)

    # 2. builder function that is returned
    def create_model():
        return GRUModel(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=DROPOUT
        )

    # create dummy model to calculate number of params
    dummy_model = create_model()
    # calculates the number of trainable params and concatenates the config into a string
    n_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    config_str = f"hidden_size:{hidden_size} | num_layers:{num_layers} | params:{n_params}"
    return create_model, config_str, n_params


def get_localformer_builder(trial, d_feat):
    hidden_size = trial.suggest_categorical("localformer_hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("localformer_num_layers", 2, 6)
    num_heads = trial.suggest_categorical("localformer_num_heads", [2, 4, 8])

    def create_model():
        return Localformer(
        d_feat=d_feat,
        d_model=hidden_size, # is equivalent to hidden_size
        nhead=num_heads,
        num_layers=num_layers,
        dropout=DROPOUT
    )

    # create dummy model to calculate number of params
    dummy_model = create_model()
    # calculates the number of trainable params and concatenates the config into a string
    n_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    config_str = f"hidden_size:{hidden_size} | num_layers:{num_layers} | num_heads:{num_heads} | params:{n_params}"

    return create_model, config_str, n_params


def get_tcn_builder(trial, d_feat):
    num_levels = trial.suggest_int("tcn_num_levels", 2, 4)
    kernel_size = trial.suggest_int("tcn_kernel_size", 2, 5)
    channel_size = trial.suggest_categorical("tcn_channel_size", [32, 64, 128])
    num_channels = [channel_size] * num_levels

    def create_model():
        return TCNModel(
        num_input=d_feat,
        num_channels=num_channels,
        output_size=1,
        kernel_size=kernel_size,
        dropout=DROPOUT
    )

    # create dummy model to calculate number of params
    dummy_model = create_model()
    # calculates the number of trainable params and concatenates the config into a string
    n_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    config_str = f"k_size:{kernel_size} | num_levels:{num_levels} | channel_size:{channel_size} | params:{n_params}"

    return create_model, config_str, n_params


def get_transformer_builder(trial, d_feat):
    hidden_size = trial.suggest_categorical("transformer_hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("transformer_num_layers", 2, 6)
    num_heads = trial.suggest_categorical("transformer_num_heads", [2, 4, 8])

    def create_model():
        return Transformer(
        d_feat=d_feat,
        d_model=hidden_size, # is equivalent to hidden_size
        nhead=num_heads,
        num_layers=num_layers,
        dropout=DROPOUT
    )

    # create dummy model to calculate number of params
    dummy_model = create_model()
    # calculates the number of trainable params and concatenates the config into a string
    n_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    config_str = f"hidden_size:{hidden_size} | num_layers:{num_layers} | num_heads:{num_heads} | params:{n_params}"

    return create_model, config_str, n_params