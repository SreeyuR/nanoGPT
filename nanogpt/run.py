"""
Based on Stanford CS224N Assignment 5 by John Hewitt <johnhew@stanford.edu> and Ansh Khurana <anshk@stanford.edu>.
Originally forked from Andrej Karpathy's minGPT.

EE148 2023SP: Assignment 3
"""

import torch
import argparse
from tqdm import tqdm

import dataset
import model
import trainer
import utils

utils.set_seed(148)

argp = argparse.ArgumentParser()
argp.add_argument('mode', help='Choose pretrain, finetune, or evaluate')
argp.add_argument('--char_corruption', action='store_true')
argp.add_argument('--reading_params_path', default=None)
argp.add_argument('--writing_params_path', default=None)
argp.add_argument('--pretrain_corpus_path', default='nanogpt/data/wiki.txt')
argp.add_argument('--finetune_corpus_path', default='nanogpt/data/birth_places_train.tsv')
argp.add_argument('--evaluate_corpus_path', default='nanogpt/data/birth_places_test.tsv')
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--pretrain_lr', default=6e-3, type=float)
argp.add_argument('--finetune_lr', default=6e-4, type=float)
args = argp.parse_args()

# save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# load the pretrain dataset
block_size = 128
pretrain_dataset = dataset.WikiDataset(
    args.char_corruption,
    block_size,
    open(args.pretrain_corpus_path, encoding='utf-8').read()
)

# DO NOT change these hyperparameters, as they're known to work
model_cfg = model.GPTConfig(
    pretrain_dataset.vocab_size,
    pretrain_dataset.block_size,
    n_layer=4,
    n_head=8,
    n_embd=256
)
gpt = model.GPT(model_cfg)
gpt.to(device)

"""
DO NOT change above here. Write your code below
"""

# Perform pretraining, finetuning, or evaluation
if args.mode == 'pretrain':
    assert args.writing_params_path is not None

    # TODO: YOUR CODE HERE #
    # [part 4e] [part 4g]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path

    # - Make sure to use the following hyperparameters for pretraining:
    # max_epochs=650
    # batch_size=128
    # learning_rate=args.pretrain_lr
    # lr_decay=True
    # warmup_tokens=512*20
    # final_tokens=200*len(pretrain_dataset)*block_size
    # num_workers=4

    # pretrain model on corpus specified
    tconf = trainer.TrainerConfig(max_epochs=650,
                                  batch_size=128,
                                  learning_rate=args.pretrain_lr,
                                  lr_decay=True,
                                  warmup_tokens=512 * 20,
                                  final_tokens=200 * len(pretrain_dataset) * block_size,
                                  # Save the resulting model in args.writing_params_path
                                  ckpt_path=args.writing_params_path,
                                  num_workers=4)
    trainer_obj = trainer.Trainer(gpt, pretrain_dataset, None, tconf)
    trainer_obj.train()
    # END OF YOUR CODE #

elif args.mode == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None

    # TODO: YOUR CODE HERE #
    # [part 4c] [part 4e] [part 4g]:
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    #
    # - Make sure to use the following hyperparameters:
    #     [part 4c] fine-tuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #     [part 4e] [part 4g] fine-tuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #
    # - You can use the args.reading_params_path flag to switch between the
    #     number of epochs for each case.

    # if a path to a pre-trained model is provided as the reading_params_path argument, load the model
    # before fine-tuning it on the birthplace prediction task
    # load finetune dataset
    finetune_dataset = dataset.NameDataset(
        pretraining_dataset=pretrain_dataset,
        data=open(args.finetune_corpus_path, encoding='utf-8').read()
    )
    # load these parameters into the model
    if args.reading_params_path is not None:
        gpt.load_state_dict(torch.load(args.reading_params_path))

    tconf = trainer.TrainerConfig(
        max_epochs=75 if args.reading_params_path is None else 10,
        batch_size=256,
        learning_rate=args.finetune_lr,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=200 * len(pretrain_dataset) * block_size,
        # Save the resulting model in args.writing_params_path
        ckpt_path=args.writing_params_path,
        num_workers=4)
    trainer_obj = trainer.Trainer(gpt, finetune_dataset, None, tconf)
    trainer_obj.train()
    # END OF YOUR CODE #


elif args.mode == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.evaluate_corpus_path is not None
    gpt.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.evaluate_corpus_path, encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)
            pred = utils.sample(gpt, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.evaluate_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
    else:
        print('Predictions written to {}; no targets provided'
              .format(args.outputs_path))
