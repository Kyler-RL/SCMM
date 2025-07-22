import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from augmentation import masking
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, f1_score, recall_score

sys.path.append("..")
warnings.filterwarnings("ignore")


# Trainer for model pre-training, fine-tuning and testing.
def Trainer(model, model_optimizer, pretrain_scheduler, finetune_scheduler, classifier, classifier_optimizer,
            pretrain_loader, finetune_loader, test_loader, configs, logger, device, experiments_logs_dir,
            training_mode):
    # Start training the SCMM model.
    logger.debug("Starting to " + training_mode + " the SCMM model.")
    best_performance = None

    # Model pre-training process.
    if training_mode == "pre_train":
        logger.debug("Pre-training on the pre-train dataset.")

        for epoch in range(1, configs.pretrain_epoch + 1):
            logger.debug(f'\nEpoch: {epoch}')
            avg_loss, avg_loss_c, avg_loss_r = \
                model_pretrain(model, model_optimizer, pretrain_scheduler, pretrain_loader, configs, device)
            logger.debug(
                "Pre-training average loss: {}, loss_c: {}, loss_r: {}".format(avg_loss, avg_loss_c, avg_loss_r))

        # Save checkpoint for the pre-trained model.
        os.makedirs(os.path.join(experiments_logs_dir, "pre_trained_model"), exist_ok=True)
        checkpoint = {"model_state_dict": model.state_dict()}
        torch.save(checkpoint, os.path.join(experiments_logs_dir, "pre_trained_model", f"ckpt_last.pt"))
        print('\nPretrained model is stored at folder: %s' % (
                experiments_logs_dir + '\\pre_trained_model' + '\\ckpt_last.pt'))

    # Model fine-tuning and testing processes.
    if training_mode == "fine_tune_test":
        logger.debug("Fine-tuning on the fine-tune dataset.")
        total_acc = []
        performance_list = []

        # Supervised fine-tuning stage.
        for epoch in range(1, configs.finetune_epoch + 1):
            logger.debug(f'\nEpoch: {epoch}')

            avg_loss, avg_acc, emb_h = model_finetune(model, model_optimizer, finetune_scheduler, finetune_loader,
                                                      logger, device, classifier, classifier_optimizer)

            # Save the best fine-tuned model according to the highest accuracy.
            if len(total_acc) == 0 or avg_acc >= max(total_acc):
                logger.debug("A better fine-tuned model is obtained, update the model parameters!")
                os.makedirs(os.path.join(experiments_logs_dir, "fine_tuned_model"), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(experiments_logs_dir, "fine_tuned_model", f"model.pt"))
                torch.save(classifier.state_dict(),
                           os.path.join(experiments_logs_dir, "fine_tuned_model", f"classifier.pt"))
            total_acc.append(avg_acc)

            # Testing.
            model.load_state_dict(torch.load(os.path.join(experiments_logs_dir, "fine_tuned_model", f"model.pt")))
            classifier.load_state_dict(
                torch.load(os.path.join(experiments_logs_dir, "fine_tuned_model", f"classifier.pt")))
            test_performance, emb_test_all, all_labels = model_test(model, test_loader, logger, device, classifier)
            performance_list.append(test_performance)

        logger.debug("\n############################## Best Testing Performance! ##############################")
        performance_array = np.array(performance_list)
        best_performance = performance_array[np.argmax(performance_array[:, 0], axis=0)]
        logger.debug(
            'Best Testing Performance: Accuracy = %.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC = %.4f '
            '| AUPRC = %.4f' % (best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                                best_performance[4], best_performance[5]))

    logger.debug("\n#################### Training is Done! ####################")
    return best_performance


# Model pre-training process.
def model_pretrain(model, optimizer, scheduler, pretrain_loader, configs, device):
    total_loss = []
    total_loss_c = []
    total_loss_r = []

    model.train()

    for batch_idx, (data, labels) in enumerate(pretrain_loader):
        mask_id, mask_data = masking(data, mask_ratio=configs.mask_ratio, mask_num=configs.mask_num,
                                     threshold=configs.threshold, mask_mode=configs.mask_mode)
        mask_data_om = torch.cat([data, mask_data], dim=0)
        data, mask_data_om = data.float().to(device), mask_data_om.float().to(device)

        loss, loss_c, loss_r = model(mask_data_om, train_mode="pre_train")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        total_loss_c.append(loss_c.item())
        total_loss_r.append(loss_r.item())

    avg_loss = torch.tensor(total_loss).mean()
    avg_loss_c = torch.tensor(total_loss_c).mean()
    avg_loss_r = torch.tensor(total_loss_r).mean()

    scheduler.step()

    return avg_loss, avg_loss_c, avg_loss_r


# Model fine-tuning process.
def model_finetune(model, optimizer, scheduler, finetune_loader, logger, device, classifier=None,
                   classifier_optimizer=None):
    global labels, pred_numpy, emb_h
    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)

    criterion = nn.CrossEntropyLoss()
    model.train()
    classifier.train()

    for batch_idx, (data, labels) in enumerate(finetune_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        emb_h = model(data, train_mode="fine_tune_test")
        predictions = classifier(emb_h)
        loss_cls = criterion(predictions, labels)

        loss_cls.backward()
        optimizer.step()
        classifier_optimizer.step()

        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
        onehot_label = F.one_hot(labels, num_classes=3).detach().cpu().numpy()
        pred_numpy = predictions.detach().cpu().numpy()
        labels_numpy = labels.detach().cpu().numpy()

        try:
            auc_bs = roc_auc_score(onehot_label, pred_numpy, average="macro", multi_class="ovr")
        except:
            auc_bs = np.float(0)
        try:
            prc_bs = average_precision_score(onehot_label, pred_numpy, average="macro")
            if not np.isnan(prc_bs):
                prc_bs = prc_bs
            else:
                prc_bs = np.float(0)
        except:
            prc_bs = np.float(0)

        if auc_bs != 0:
            total_auc.append(auc_bs)
        if prc_bs != 0:
            total_prc.append(prc_bs)

        total_acc.append(acc_bs)
        total_loss.append(loss_cls.item())

        labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
        pred_numpy = np.argmax(pred_numpy, axis=1)
        pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))

    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro')
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro')
    F1_Score = f1_score(labels_numpy_all, pred_numpy_all, average='macro')

    avg_loss = torch.tensor(total_loss).mean()
    avg_acc = torch.tensor(total_acc).mean()
    avg_auc = torch.tensor(total_auc).mean()
    avg_prc = torch.tensor(total_prc).mean()

    scheduler.step(avg_loss)

    logger.debug("Fine-tuning average loss: {}".format(avg_loss))
    print('Fine-tuning: Accuracy = %.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC = %.4f | AUPRC = %.4f'
          % (avg_acc * 100, precision * 100, recall * 100, F1_Score * 100, avg_auc * 100, avg_prc * 100))
    return avg_loss, avg_acc, emb_h


# Model testing process.
def model_test(model, test_loader, logger, device, classifier=None):
    total_acc = []
    total_auc = []
    total_prc = []
    emb_test_all = []

    model.eval()
    classifier.eval()

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)

        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.float().to(device), labels.long().to(device)
            emb_h = model(data, train_mode="fine_tune_test")
            emb_test_all.append(emb_h)

            predictions_test = classifier(emb_h)
            acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(labels, num_classes=3).detach().cpu().numpy()
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()

            try:
                auc_bs = roc_auc_score(onehot_label, pred_numpy, average="macro", multi_class="ovr")
            except:
                auc_bs = np.float(0)
            try:
                prc_bs = average_precision_score(onehot_label, pred_numpy, average="macro")
                if not np.isnan(prc_bs):
                    prc_bs = prc_bs
                else:
                    prc_bs = np.float(0)
            except:
                prc_bs = np.float(0)

            if auc_bs != 0:
                total_auc.append(auc_bs)
            if prc_bs != 0:
                total_prc.append(prc_bs)

            total_acc.append(acc_bs)

            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy = np.argmax(pred_numpy, axis=1)
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))

    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]
    all_labels = torch.Tensor([labels_numpy_all, pred_numpy_all])

    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro')
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro')
    F1_Score = f1_score(labels_numpy_all, pred_numpy_all, average='macro')

    avg_acc = torch.tensor(total_acc).mean()
    avg_auc = torch.tensor(total_auc).mean()
    avg_prc = torch.tensor(total_prc).mean()

    test_performance = [avg_acc * 100, precision * 100, recall * 100, F1_Score * 100, avg_auc * 100, avg_prc * 100]
    emb_test_all = torch.concat(tuple(emb_test_all), dim=0)

    logger.debug('Testing: Accuracy = %.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC = %.4f | AUPRC = %.4f'
                 % (avg_acc * 100, precision * 100, recall * 100, F1_Score * 100, avg_auc * 100, avg_prc * 100))
    return test_performance, emb_test_all, all_labels
