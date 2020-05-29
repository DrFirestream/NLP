import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo
LABEL_LIST = ["negative", "neutral", "positive"]
LANG_MODEL = "bert-base-german-cased"
tokenizer = Tokenizer.load(    LANG_MODEL,    do_lower_case=False)
processor = TextClassificationProcessor(    tokenizer=tokenizer,    max_seq_len=128,    data_dir='.', train_filename='train.tsv', dev_filename='dev.tsv', test_filename='test2.tsv',    label_list=LABEL_LIST,    label_column_name="sentiment",    metric="acc")
data_silo = DataSilo(    processor=processor,    batch_size=90)
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead,CapsuleTextClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
language_model = LanguageModel.load(LANG_MODEL)
#prediction_head = TextClassificationHead(num_labels=len(LABEL_LIST))
prediction_head = CapsuleTextClassificationHead(len(LABEL_LIST), d_depth=14,d_inp=64, n_parts=64, d_cap=2)
sentiment_model = AdaptiveModel(    language_model=language_model,    prediction_heads=[prediction_head],    embeds_dropout_prob=0.1,    lm_output_types=["per_sequence"],    device="cuda")
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from pathlib import Path
sentiment_model, optimizer, lr_schedule = initialize_optimizer(    model=sentiment_model,    learning_rate=1e-3,    device="cuda",    n_batches=len(data_silo.loaders["train"]),    n_epochs=3)
trainer = Trainer(   model=sentiment_model, n_gpu = 1,    optimizer=optimizer,    data_silo=data_silo,    epochs=3,    lr_schedule=lr_schedule,    evaluate_every=232, device="cuda", checkpoint_every=232, checkpoint_root_dir=Path('./ckpt2'), freeze = True)
trainer.train()

trainer = Trainer.create_or_load_checkpoint(data_silo = data_silo, checkpoint_root_dir=Path('./ckpt2'), resume_from_checkpoint='epoch_2_step_232')
trainer.epochs = 6
sentiment_model, optimizer, lr_schedule = initialize_optimizer(    model=trainer.model,    learning_rate=1e-4,    device="cuda",    n_batches=len(data_silo.loaders["train"]),    n_epochs=6,  optimizer_opts={"name": "TransformersAdamW", "correct_bias": True, "weight_decay": 0.0})
trainer.model = sentiment_model
trainer.optimizer = optimizer
trainer.lr_schedule = lr_schedule 
trainer.freeze = False
trainer.train()
