import torch
from datasets import load_dataset
from transformers import (
    ResNetForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    ResNetConfig,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    Resize,
    CenterCrop,
    RandomResizedCrop,
    RandomHorizontalFlip,
)

def train():
    # 1. Load Dataset
    print("Loading CIFAR-10 dataset...")
    dataset = load_dataset("cifar10")
    
    # 2. Preprocessing
    # ResNet standard normalization
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Define transforms
    # We'll upsample slightly to 224x224 to match standard ResNet expectations better,
    # or we can use a smaller model configuration. 
    # For "cheap" execution, let's try to keep it small, but standard ResNet-18 
    # usually expects larger inputs. 
    # However, to keep it FAST on a laptop, let's stick to 32x32 or slightly larger if needed.
    # Hugging Face's ResNet implementation handles dynamic sizes well, but let's resize to 64x64
    # to be safe and get decent features without too much compute.
    size = 64 
    
    _train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    
    _val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    def train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(image.convert("RGB")) for image in examples["img"]]
        del examples["img"]
        return examples

    def val_transforms(examples):
        examples["pixel_values"] = [_val_transforms(image.convert("RGB")) for image in examples["img"]]
        del examples["img"]
        return examples

    print("Preprocessing dataset...")
    # Split train into train/val (CIFAR-10 has 'train' and 'test', we'll use 'test' as validation for simplicity here)
    train_ds = dataset["train"]
    val_ds = dataset["test"]

    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)

    # 3. Model
    print("Initializing ResNet-18...")
    # We use a configuration that fits our number of labels
    # We can load a pre-trained one and ignore the size mismatch warnings, 
    # or initialize from scratch. Initializing from scratch is "cheaper" in terms of 
    # not downloading a huge model, but training takes longer to converge.
    # Given the request for "checkpoints at each epoch", training from scratch 
    # makes the evolution more interesting to visualize later.
    
    labels = dataset["train"].features["label"].names
    config = ResNetConfig.from_pretrained("microsoft/resnet-18")
    config.num_labels = len(labels)
    
    # Adjusting downsampling to be friendlier to small images if we were doing 32x32,
    # but with 64x64 standard config should be okay-ish.
    # Actually, let's just use the standard config but initialize weights randomly 
    # or use pretrained=True and fine-tune. Fine-tuning is faster.
    
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-18",
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )

    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    args = parser.parse_args()

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./cifar10_resnet18_checkpoints",
        per_device_train_batch_size=32, # Adjust based on RAM
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs, # Use command line argument
        save_strategy="epoch", # CRITICAL: Save every epoch
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        learning_rate=2e-4,
        save_total_limit=None, # Keep all checkpoints
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none", # Disable wandb etc for now
        load_best_model_at_end=False, # We want the trajectory, not just the best
        dataloader_num_workers=0, # Avoid multiprocessing issues on Mac sometimes
        use_mps_device=True if torch.backends.mps.is_available() else False, # Explicitly enable MPS if needed/supported
    )

    from transformers import TrainerCallback
    import os
    import shutil

    class RenameCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # Get the latest checkpoint path
            if os.path.isdir(args.output_dir):
                checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    # Sort by modification time to get the latest
                    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(args.output_dir, x)))
                    latest_checkpoint_path = os.path.join(args.output_dir, latest_checkpoint)
                    
                    # Construct new name
                    epoch = int(state.epoch)
                    new_name = f"resnet18_cifar10_epoch_{epoch}"
                    new_path = os.path.join(args.output_dir, new_name)
                    
                    # Rename
                    if not os.path.exists(new_path):
                        print(f"Renaming checkpoint {latest_checkpoint} to {new_name}")
                        os.rename(latest_checkpoint_path, new_path)
                    else:
                        print(f"Checkpoint {new_name} already exists, skipping rename.")

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=None, # Not needed for vision
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[RenameCheckpointCallback()],
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

def compute_metrics(eval_pred):
    import numpy as np
    import numpy as np
    import evaluate
    
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    train()
