from transformers import AutoImageProcessor
from utils import lab2idx, idx2lab
from dataset import SyntheticData, collate_fn
from transformers import AutoModelForObjectDetection, AutoConfig
from transformers import TrainingArguments
from transformers import set_seed
from transformers import Trainer
import argparse
import wandb

def main(checkpoint, training_args, train_data_path, val_data_path, resume_from_checkpoint):
    image_size = 480
    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"width": image_size, "height": image_size},
    )

    config = AutoConfig.from_pretrained(checkpoint)
    config.id2label = idx2lab
    config.label2id = lab2idx

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )

    set_seed(training_args.seed)

    train_dataset = SyntheticData(train_data_path, image_processor)
    validation_dataset = SyntheticData(val_data_path, image_processor)

    # wandb.init(
    # project="multimodal",  # Set your project name here
    # name=args.run_name,
    # id=args.wandb_run_id,
    # resume="allow" if resume_from_checkpoint else None,
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
    )

    # print("WandB Run ID:", wandb.run.id)
    # with open("wandb_run_id.txt", "w") as f:
    #     f.write(wandb.run.id)

    trainer.train(resume_from_checkpoint=checkpoint if resume_from_checkpoint else None)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DAB-DETR model")
    parser.add_argument("--checkpoint", type=str, default="IDEA-Research/dab-detr-resnet-50-dc5", help="Model checkpoint")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--run_name", type=str, default="finetune-dab-detr", help="WandB run name")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Maximum gradient norm")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at the end of training")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Total number of saved checkpoints")
    parser.add_argument("--remove_unused_columns", action="store_true", help="Remove unused columns")
    parser.add_argument("--eval_do_concat_batches", action="store_true", help="Concatenate evaluation batches")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool (wandb)")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="Find unused parameters in DDP")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID to resume logging to")

    return parser.parse_args()


def get_training_args(args):
    return TrainingArguments(
        run_name=args.run_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=args.load_best_model_at_end,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=args.remove_unused_columns,
        eval_do_concat_batches=args.eval_do_concat_batches,
        report_to=args.report_to,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        seed=args.seed,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )

if __name__ == "__main__":

    args = parse_args()
    training_args = get_training_args(args)
    checkpoint = args.checkpoint
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    resume_from_checkpoint = args.resume_from_checkpoint

    print("Resume from checkpoint: ", resume_from_checkpoint)
    
    main(checkpoint, training_args, train_data_path, val_data_path, resume_from_checkpoint) 