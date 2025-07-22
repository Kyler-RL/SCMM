import os
import argparse
import torch.backends.cudnn

from datetime import datetime
from utils import _logger
from data_loader import data_generator
from model import *
from trainer import Trainer
from config_files.SEED_IV_Configs import Config

parser = argparse.ArgumentParser()
parser.add_argument("--run_description", default="run1", type=str, help="Number of experiments")
parser.add_argument("--seed", default=42, type=int, help="Seed value")
parser.add_argument("--training_mode", default="pre_train", type=str, help="pre_train, fine_tune_test")
parser.add_argument("--pretrain_dataset", default="SEED_IV", type=str, help="SEED, SEED_IV, DEAP")
parser.add_argument("--finetune_dataset", default="SEED", type=str, help="SEED, SEED_IV, DEAP")
parser.add_argument("--logs_save_dir", default="..\\experiments_logs", type=str, help="Saving directory")
parser.add_argument("--device", default="cuda", type=str, help="CPU or GPU")
args, unknown = parser.parse_known_args()


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    return seed


with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s to train the SCMM model now.' % device)


def main(args, random_seed=None):
    start_time = datetime.now()
    method = "SCMM"
    pretrain_dataset = args.pretrain_dataset
    finetune_dataset = args.finetune_dataset
    experiment_description = str(pretrain_dataset) + '_2_' + str(finetune_dataset)
    training_mode = args.training_mode
    run_description = args.run_description
    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)
    exec(f'from config_files.{pretrain_dataset}_Configs import Config as Configs')
    configs = Config()

    if random_seed is not None:
        SEED = set_random_seed(random_seed)
    else:
        SEED = set_random_seed(args.seed)

    experiments_logs_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                        training_mode + f"_3layer_1D_CNN")
    os.makedirs(experiments_logs_dir, exist_ok=True)

    log_file_name = os.path.join(experiments_logs_dir, f"logs_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 50)
    logger.debug(f'Pre-training Dataset: {pretrain_dataset}')
    logger.debug(f'Fine-tuning Dataset:  {finetune_dataset}')
    logger.debug(f'Random seed:  {SEED}')
    logger.debug(f'Model:        {method}')
    logger.debug(f'Mode:         {training_mode}')
    logger.debug("=" * 50)

    source_path = f"../datasets/{pretrain_dataset}/3_class/session_1/all_subject/"
    target_path = f"../datasets/{finetune_dataset}/3_class/session_1/single_subject/sub_1/trial_based/9_trials/"
    normalize = True
    subset = False

    logger.debug("Loading Datasets ...")
    pretrain_loader, finetune_loader, test_loader = \
        data_generator(source_path, target_path, configs, normalize=normalize, subset=subset)

    model = SCMM(configs).to(device)
    classifier = target_classifier(configs).to(device)

    if training_mode == "fine_tune_test":
        load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
                                              f"pre_train_3layer_1D_CNN", "pre_trained_model"))
        print("Loading pre-trained SCMM model from: ", load_from)
        ckpt = torch.load(os.path.join(load_from, "ckpt_last.pt"), map_location=device)
        pretrained_dict = ckpt["model_state_dict"]
        model.load_state_dict(pretrained_dict)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate,
                                       betas=(configs.beta1, configs.beta2), weight_decay=configs.weight_decay)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.learning_rate,
                                            betas=(configs.beta1, configs.beta2), weight_decay=configs.weight_decay)
    pretrain_scheduler = \
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=configs.pretrain_epoch)
    finetune_scheduler = \
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=configs.finetune_epoch)

    best_performance = Trainer(model, model_optimizer, pretrain_scheduler, finetune_scheduler,
                               classifier, classifier_optimizer, pretrain_loader, finetune_loader, test_loader,
                               configs, logger, device, experiments_logs_dir, training_mode)

    logger.debug(f"Training time is: {datetime.now() - start_time}")


if __name__ == "__main__":
    main(args)
