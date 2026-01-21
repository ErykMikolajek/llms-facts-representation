from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import find_device
import dataset_sequencing
import activations_collecting
import autoencoder_training
import features_analysis
import os

MODEL_NAME = 'roneneldan/TinyStories-1M'
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
DATASET_PATH = "data/tinystories_dataset"

SEQ_LENGTH = 256  # Max sequence length for model
LAYER_NUM = 4

BATCH_SIZE = 4  # LLM activation collection batch size
DS_FRACTION = .01
INTERACTIVE = True


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    device = find_device()
    print(f"Using device: {device}")

    print(10*"=", "Preparing sequences for activation collection...", 10*"=")
    dataset_sequencing.prepare_sequences(
        data_path=DATASET_PATH,
        tokenizer=tokenizer,
        seq_length=SEQ_LENGTH,
        min_seq_length=10,  # TODO: parametrize this
        batch_sentences=10, # TODO: parametrize this
        ds_fraction=DS_FRACTION,
        interactive=INTERACTIVE
    )

    print(10*"=", "Collecting activations...", 10*"=")
    activations_collecting.collect_activations(
        model=model,
        tokenizer=tokenizer,
        data_path=DATASET_PATH,
        seq_length=SEQ_LENGTH,
        layer_num=LAYER_NUM,
        batch_size=BATCH_SIZE,
    )

    print(10*"=", "Training autoencoder...", 10*"=")
    sae = autoencoder_training.train_autoencoder(
        d_model=64,
        expansion_factor=64,
        k=8,
        data_path=DATASET_PATH,
        seq_length=SEQ_LENGTH,
        layer_num=LAYER_NUM,
        batch_size_sae=256,
        num_epochs=5,
        learning_rate=1e-3,
        save_every_n_steps=1000
    )

    print(10*"=", "Analyzing features...", 10*"=")
    features_analysis.analyze_sae(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        path_dir=DATASET_PATH,
        layer_num=LAYER_NUM,
    )
