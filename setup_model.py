import copy
import math
import os
import pickle
from time import time
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from scipy.stats import gmean
from sklearn.metrics import f1_score, accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score, recall_score, roc_curve, confusion_matrix
from torchvision.utils import make_grid

# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)


class import_and_train_model:
    def __init__(self):

        return



    def import_deit_models_for_testing(self, train_main, test_main,testing=False):
        classes = np.load(test_main.main_param_path + '/classes.npy')
        num_classes=len(np.unique(classes))

        if train_main.architecture == 'deit':
            self.model = timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'efficientnetb2':
            self.model = timm.create_model('tf_efficientnet_b2.ns_jft_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'efficientnetb5':
            self.model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'efficientnetb6':
            self.model = timm.create_model('tf_efficientnet_b6.ns_jft_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'efficientnetb7':
            self.model = timm.create_model('tf_efficientnet_b7.ns_jft_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'densenet':
            self.model = timm.create_model('densenet161.tv_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'mobilenet':
            self.model = timm.create_model('mobilenetv3_large_100.miil_in21k_ft_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'inception':
            self.model = timm.create_model('inception_v4.tf_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'vit':
            self.model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'mae':
            self.model = timm.create_model('vit_base_patch16_224.mae', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'swin':
            self.model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        elif train_main.architecture == 'beit':
            self.model = timm.create_model('beit_base_patch16_224.in22k_ft_in22k_in1k', pretrained=not testing,
                                           num_classes=num_classes)
        else:
            print('This model cannot be imported. Please check from the list of models')

        # additional layers
        if train_main.add_layer == 'yes':
            if train_main.architecture == 'deit':
                in_features = self.model.get_classifier()[-1].in_features
                pretrained_layers = list(self.model.children())[:-2]
                additional_layers = nn.Sequential(
                                        nn.Dropout(p=train_main.dropout_1),
                                        nn.Linear(in_features=in_features, out_features=train_main.fc_node),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=train_main.dropout_2),
                                        nn.Linear(in_features=train_main.fc_node, out_features=num_classes),
                                        )
                self.model = nn.Sequential(*pretrained_layers, additional_layers)

            else:
                in_features = self.model.get_classifier().in_features
                pretrained_layers = list(self.model.children())[:-1]
                additional_layers = nn.Sequential(
                                        nn.Dropout(p=train_main.dropout_1),
                                        nn.Linear(in_features=in_features, out_features=train_main.fc_node),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=train_main.dropout_2),
                                        nn.Linear(in_features=train_main.fc_node, out_features=num_classes),
                                        )
                self.model = nn.Sequential(*pretrained_layers, additional_layers)


        if torch.cuda.is_available() and test_main.use_gpu == 'yes':
            device = torch.device("cuda:" + str(test_main.gpu_id))
        else:
            device = torch.device("cpu")

        # model = nn.DataParallel(model)  # to run on multiple GPUs
        self.model.to(device)

        if train_main.add_layer == 'yes':
            if train_main.last_layer_finetune == 'yes':
                n_layer = 0
                for param in self.model.parameters():
                    n_layer += 1
                    param.requires_grad = False

                for i, param in enumerate(self.model.parameters()):
                    if i + 1 > n_layer - 5:
                        param.requires_grad = True

            else:
                for param in self.model.parameters():
                    param.requires_grad = True

        elif train_main.add_layer == 'no':
            if train_main.last_layer_finetune == 'yes':
                n_layer = 0
                for param in self.model.parameters():
                    n_layer += 1
                    param.requires_grad = False

                for i, param in enumerate(self.model.parameters()):
                    if i + 1 > n_layer - 2:
                        param.requires_grad = True

            else:
                for param in self.model.parameters():
                    param.requires_grad = True

        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

        class_weights_tensor = torch.load(test_main.main_param_path + '/class_weights_tensor.pt')
        self.criterion = nn.CrossEntropyLoss(class_weights_tensor)
        if torch.cuda.is_available() and test_main.use_gpu == 'yes':
            torch.cuda.set_device(test_main.gpu_id)
            self.model.cuda(test_main.gpu_id)
            self.criterion = self.criterion.cuda(test_main.gpu_id)


    def run_prediction(self, train_main, data_loader, name):
        # classes = np.load(train_main.outpath + '/classes.npy')
        classes = data_loader.classes
        PATH = data_loader.checkpoint_path + '/trained_model_' + name + '.pth'
        im_names = data_loader.filenames

        with open(data_loader.checkpoint_path + '/file_names_' + name + '.pickle', 'wb') as b:
            pickle.dump(im_names, b)

        if torch.cuda.is_available() and train_main.use_gpu == 'yes':
            checkpoint = torch.load(PATH, map_location="cuda:" + str(train_main.gpu_id))
        else:
            checkpoint = torch.load(PATH, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        avg_acc1, target, output, prob = cls_predict(train_main, data_loader.test_dataloader,
                                                     self.model,
                                                     self.criterion,
                                                     time_begin=time())
        target = torch.cat(target)
        output = torch.cat(output)
        prob = torch.cat(prob)

        target = target.cpu().numpy()
        output = output.cpu().numpy()
        prob = prob.cpu().numpy()

        output_max = output.argmax(axis=1)

        target_label = np.array([classes[target[i]] for i in range(len(target))], dtype=object)
        output_label = np.array([classes[output_max[i]] for i in range(len(output_max))], dtype=object)

        GT_Pred_GTLabel_PredLabel_Prob = [target, output_max, target_label, output_label, prob]
        with open(data_loader.checkpoint_path + '/GT_Pred_GTLabel_PredLabel_prob_model_' + name + '.pickle', 'wb') \
                as cw:
            pickle.dump(GT_Pred_GTLabel_PredLabel_Prob, cw)

        accuracy_model = accuracy_score(target_label, output_label)
        clf_report = classification_report(target_label, output_label)
        clf_report_rm_0 = classification_report(target_label, output_label, labels=np.unique(target_label))
        f1 = f1_score(target_label, output_label, average='macro')
        f1_rm_0 = f1_score(target_label, output_label, average='macro', labels=np.unique(target_label))

        f = open(data_loader.checkpoint_path + 'test_report_' + name + '.txt', 'w')
        f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                              clf_report))
        f.close()

        ff = open(data_loader.checkpoint_path + 'test_report_rm_0_' + name + '.txt', 'w')
        ff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1_rm_0,
                                                                                              clf_report_rm_0))
        ff.close()

    def load_trained_model(self, train_main, data_loader, modeltype):
        # self.import_deit_models(train_main, data_loader)

        if modeltype == 0:
            PATH = data_loader.checkpoint_path + 'trained_model_original.pth'
        elif modeltype == 1:
            PATH = data_loader.checkpoint_path + 'trained_model_tuned.pth'
        elif modeltype == 2:
            PATH = data_loader.checkpoint_path + 'trained_model_finetuned.pth'

        if torch.cuda.is_available() and train_main.use_gpu == 'yes':
            checkpoint = torch.load(PATH, map_location="cuda:" + str(train_main.gpu_id))
        else:
            checkpoint = torch.load(PATH, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        self.f1 = checkpoint['f1']
        self.acc = checkpoint['acc']
        self.initial_epoch = checkpoint['epoch']
        self.best_values = [self.loss, self.f1, self.acc]


    def init_train_predict(self, train_main, data_loader, modeltype):
        if modeltype == 0:
            self.initialize_model(train_main=train_main, test_main=None,
                                  data_loader=data_loader, lr=train_main.lr)
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.epochs,
                              train_main.lr, "original", self.best_values, modeltype)
            self.run_prediction(train_main, data_loader, 'original')

        elif modeltype == 1:
            self.initialize_model(train_main=train_main, test_main=None,
                                  data_loader=data_loader, lr=train_main.finetune_lr)

            if train_main.last_layer_finetune_1 == 'yes':
                n_layer = 0
                for param in self.model.parameters():
                    n_layer += 1
                    param.requires_grad = False

                for i, param in enumerate(self.model.parameters()):
                    if i + 1 > n_layer - 5:
                        param.requires_grad = True

            else:
                for param in self.model.parameters():
                    param.requires_grad = True

            total_trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"{total_trainable_params:,} training parameters.")

            self.run_training(train_main, data_loader, self.initial_epoch, train_main.finetune_epochs,
                              train_main.finetune_lr, "tuned", self.best_values, modeltype)
            self.run_prediction(train_main, data_loader, 'tuned')

        elif modeltype == 2:
            self.initialize_model(train_main=train_main, test_main=None,
                                  data_loader=data_loader, lr=train_main.finetune_lr / 10)

            if train_main.last_layer_finetune_2 == 'yes':
                n_layer = 0
                for param in self.model.parameters():
                    n_layer += 1
                    param.requires_grad = False

                for i, param in enumerate(self.model.parameters()):
                    if i + 1 > n_layer - 5:
                        param.requires_grad = True

            else:
                for param in self.model.parameters():
                    param.requires_grad = True

            total_trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"{total_trainable_params:,} training parameters.")

            self.run_training(train_main, data_loader, self.initial_epoch, train_main.finetune_epochs,
                              train_main.finetune_lr / 10, "finetuned", self.best_values, modeltype)
            self.run_prediction(train_main, data_loader, 'finetuned')

    def train_predict(self, train_main, data_loader, modeltype):
        if modeltype == 0:
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.epochs,
                              train_main.lr, "original", self.best_values, modeltype)
            self.run_prediction(train_main, data_loader, 'original')

        elif modeltype == 1:
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.finetune_epochs,
                              train_main.finetune_lr, "tuned", self.best_values, modeltype)
            self.run_prediction(train_main, data_loader, 'tuned')

        elif modeltype == 2:
            self.run_training(train_main, data_loader, self.initial_epoch, train_main.finetune_epochs,
                              train_main.finetune_lr / 10, "finetuned", self.best_values, modeltype)
            self.run_prediction(train_main, data_loader, 'finetuned')

    def train_and_save(self, train_main, data_loader):
        model_present_path0 = data_loader.checkpoint_path + 'trained_model_original.pth'
        model_present_path1 = data_loader.checkpoint_path + 'trained_model_tuned.pth'
        model_present_path2 = data_loader.checkpoint_path + 'trained_model_finetuned.pth'

        self.import_deit_models(train_main, data_loader)

        if train_main.resume_from_saved == 'no':
            if train_main.finetune == 2:
                if not os.path.exists(model_present_path0):
                    self.train_predict(train_main, data_loader, 0)
                    self.init_train_predict(train_main, data_loader, 1)
                    self.init_train_predict(train_main, data_loader, 2)
                elif not os.path.exists(model_present_path1):
                    print(' I am using trained_model_original.pth as the base')
                    self.load_trained_model(train_main, data_loader, 0)
                    print('Now training tuned model')
                    self.initial_epoch = 0
                    self.init_train_predict(train_main, data_loader, 1)
                    print('Now training finetuned model')
                    self.initial_epoch = 0
                    self.init_train_predict(train_main, data_loader, 2)
                elif not os.path.exists(model_present_path2):
                    print(' I am using trained_model_tuned.pth as the base')
                    self.load_trained_model(train_main, data_loader, 1)
                    print('Now training finetuned model')
                    self.initial_epoch = 0
                    self.init_train_predict(train_main, data_loader, 2)
                else:
                    print('If you want to retrain then set "resume from saved" to "yes"')
                    self.run_prediction(train_main, data_loader, 'finetuned')

            elif train_main.finetune == 1:
                if not os.path.exists(model_present_path0):
                    self.train_predict(train_main, data_loader, 0)
                    self.init_train_predict(train_main, data_loader, 1)
                elif not os.path.exists(model_present_path1):
                    print(' I am using trained_model_original.pth as the base')
                    self.load_trained_model(train_main, data_loader, 0)
                    print('Now training tuned model')
                    self.initial_epoch = 0
                    self.init_train_predict(train_main, data_loader, 1)
                else:
                    print('If you want to retrain then set "resume from saved" to "yes"')
                    self.run_prediction(train_main, data_loader, 'tuned')

            elif train_main.finetune == 0:
                if not os.path.exists(model_present_path0):
                    self.train_predict(train_main, data_loader, 0)
                else:
                    print('If you want to retrain then set "resume from saved" to "yes"')
                    self.run_prediction(train_main, data_loader, 'original')

        elif train_main.resume_from_saved == 'yes':
            if train_main.finetune == 0:
                if not os.path.exists(model_present_path0):
                    self.train_predict(train_main, data_loader, 0)
                else:
                    self.resuming_training(train_main, data_loader, 0)

            elif train_main.finetune == 1:
                if not os.path.exists(model_present_path0):
                    self.train_predict(train_main, data_loader, 0)
                    self.init_train_predict(train_main, data_loader, 1)
                elif not os.path.exists(model_present_path1):
                    print(' I am using trained_model_original.pth as the base')
                    self.resuming_training(train_main, data_loader, 0)
                    print('Now training tuned model')
                    self.initial_epoch = 0
                    self.init_train_predict(train_main, data_loader, 1)
                else:
                    print(' I am using trained_model_tuned.pth as the base')
                    self.resuming_training(train_main, data_loader, 1)

            elif train_main.finetune == 2:

                if not os.path.exists(model_present_path0):
                    self.train_predict(train_main, data_loader, 0)
                    self.init_train_predict(train_main, data_loader, 1)
                    self.init_train_predict(train_main, data_loader, 2)
                elif not os.path.exists(model_present_path1):
                    print(' I am using trained_model_original.pth as the base')
                    self.resuming_training(train_main, data_loader, 0)
                    print('Now training tuned model')
                    self.initial_epoch = 0
                    self.init_train_predict(train_main, data_loader, 1)
                    print('Now training finetuned model')
                    self.initial_epoch = 0
                    self.init_train_predict(train_main, data_loader, 2)

                elif not os.path.exists(model_present_path2):
                    print(' I am using trained_model_tuned.pth as the base')
                    self.resuming_training(train_main, data_loader, 1)
                    print('Now training finetuned model')
                    self.initial_epoch = 0
                    self.init_train_predict(train_main, data_loader, 2)
                else:
                    print(' I am using trained_model_finetuned.pth as the base')
                    self.resuming_training(train_main, data_loader, 2)
            else:
                print('Choose the correct finetune label')

    def run_prediction_on_unseen(self, train_main, test_main, data_loader, name):
        classes = np.load(test_main.main_param_path + '/classes.npy')
        if len(test_main.model_path) > 1:
            print("Do you want to predict using ensemble model ? If so then set the ensemble parameter to 1 and run "
                  "again")
        else:
            checkpoint_path = test_main.model_path[0]
            PATH = checkpoint_path + '/trained_model_' + name + '.pth'
            # PATH = data_loader.checkpoint_path + '/trained_model_' + name + '.pth'
            im_names = data_loader.filenames

            if torch.cuda.is_available() and test_main.use_gpu == 'yes':
                checkpoint = torch.load(PATH, map_location="cuda:" + str(test_main.gpu_id))
            else:
                checkpoint = torch.load(PATH, map_location='cpu')

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # device = torch.device("cpu")
            # self.model = self.model.module.to(device)

            output, prob = cls_predict_on_unseen(train_main, test_main, data_loader.test_dataloader, self.model)

            output = torch.cat(output)
            prob = torch.cat(prob)

            output = output.cpu().numpy()
            prob = prob.cpu().numpy()

            output_max = output.argmax(axis=1)

            output_label = np.array([classes[output_max[i]] for i in range(len(output_max))], dtype=object)

            output_corrected_label = copy.deepcopy(output_label)

            first_indices = prob.argsort()[:, -1]
            confs = [prob[i][first_indices[i]] for i in range(len(first_indices))]
            for i in range(len(confs)):
                if confs[i] < test_main.threshold:
                    output_corrected_label[i] = 'unknown'

            # Pred_PredLabel_Prob = [output_max, output_label, output_corrected_label, prob]
            # with open(test_main.test_outpath + '/Single_model_Pred_PredLabel_Prob_' + name + '.pickle',
            # 'wb') as cw:
            #     pickle.dump(Pred_PredLabel_Prob, cw)

            output_label = output_label.tolist()

            if test_main.threshold > 0:
                print('I am using threshold value as : {}'.format(test_main.threshold))
                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_label)]
                np.savetxt(test_main.test_outpath + '/Single_model_Plankiformer_predictions.txt', To_write,
                           fmt='%s')

                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_corrected_label)]
                np.savetxt(test_main.test_outpath + '/Single_model_Plankiformer_predictions_thresholded.txt',
                           To_write, fmt='%s')
            else:
                print('I am using default value as threshold i.e. 0')
                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_label)]
                np.savetxt(test_main.test_outpath + '/Single_model_Plankiformer_predictions.txt', To_write,
                           fmt='%s')

    def run_ensemble_prediction_on_unseen(self, train_main, test_main, data_loader, name):
        classes = np.load(test_main.main_param_path + '/classes.npy')
        Ensemble_prob = []
        im_names = data_loader.Filenames

        if test_main.TTA == 'no':
            for i in range(len(test_main.model_path)):
                checkpoint_path = test_main.model_path[i]
                PATH = checkpoint_path + '/trained_model_' + name + '.pth'
                if torch.cuda.is_available() and test_main.use_gpu == 'yes':
                    checkpoint = torch.load(PATH, map_location="cuda:" + str(test_main.gpu_id))
                else:
                    checkpoint = torch.load(PATH, map_location='cpu')

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # device = torch.device("cpu")
                # self.model = self.model.module.to(device)

                output, prob = cls_predict_on_unseen(train_main, test_main, data_loader.test_dataloader, self.model)

                prob = torch.cat(prob)

                prob = prob.cpu().numpy()

                Ensemble_prob.append(prob)

        if test_main.TTA == 'yes':
            for i in range(len(test_main.model_path)):
                checkpoint_path = test_main.model_path[i]
                PATH = checkpoint_path + '/trained_model_' + name + '.pth'
                if torch.cuda.is_available() and test_main.use_gpu == 'yes':
                    checkpoint = torch.load(PATH, map_location="cuda:" + str(test_main.gpu_id))
                else:
                    checkpoint = torch.load(PATH, map_location='cpu')

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # device = torch.device("cpu")
                # self.model = self.model.module.to(device)

                output, prob = cls_predict_on_unseen(train_main, test_main, data_loader.test_dataloader, self.model)
                output_1, prob_1 = cls_predict_on_unseen(train_main, test_main, data_loader.test_dataloader_1, self.model)
                output_2, prob_2 = cls_predict_on_unseen(train_main, test_main, data_loader.test_dataloader_2, self.model)
                output_3, prob_3 = cls_predict_on_unseen(train_main, test_main, data_loader.test_dataloader_3, self.model)

                prob = torch.cat(prob)
                prob_1 = torch.cat(prob_1)
                prob_2 = torch.cat(prob_2)
                prob_3 = torch.cat(prob_3)

                prob = prob.cpu().numpy()
                prob_1 = prob_1.cpu().numpy()
                prob_2 = prob_2.cpu().numpy()
                prob_3 = prob_3.cpu().numpy()

                Ensemble_prob.append(prob)
                Ensemble_prob.append(prob_1)
                Ensemble_prob.append(prob_2)
                Ensemble_prob.append(prob_3)

        Ens_DEIT_prob_max = []
        Ens_DEIT_label = []
        Ens_DEIT = []
        name2 = []

        if test_main.ensemble == 1:
            Ens_DEIT = sum(Ensemble_prob) / len(Ensemble_prob)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name2 = 'arth_mean_'

        elif test_main.ensemble == 2:
            Ens_DEIT = gmean(Ensemble_prob)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name2 = 'geo_mean_'

        Ens_DEIT_corrected_label = copy.deepcopy(Ens_DEIT_label)

        first_indices = Ens_DEIT.argsort()[:, -1]
        Ens_confs = [Ens_DEIT[i][first_indices[i]] for i in range(len(first_indices))]

        for i in range(len(Ens_confs)):
            if Ens_confs[i] < test_main.threshold:
                Ens_DEIT_corrected_label[i] = 'unknown'

        # Pred_PredLabel_Prob = [Ens_DEIT_prob_max, Ens_DEIT_label, Ens_DEIT_corrected_label, Ens_DEIT]
        # with open(test_main.test_outpath + '/Ensemble_models_Pred_PredLabel_Prob_' + name2 + name + '.pickle',
        #           'wb') as cw:
        #     pickle.dump(Pred_PredLabel_Prob, cw)

        Ens_DEIT_label = Ens_DEIT_label.tolist()

        if test_main.threshold > 0:
            print('I am using threshold value as : {}'.format(test_main.threshold))
            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_label)]
            np.savetxt(test_main.test_outpath + '/Ensemble_models_Plankiformer_predictions_' + name2 + name +
                       '.txt', To_write, fmt='%s')

            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_corrected_label)]
            np.savetxt(test_main.test_outpath + '/Ensemble_models_Plankiformer_predictions_' + name2 + name +
                       '_thresholded.txt', To_write, fmt='%s')
        else:
            print('I am using default value as threshold i.e. 0')
            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_label)]
            np.savetxt(test_main.test_outpath + '/Ensemble_models_Plankiformer_predictions_' + name2 + name +
                       '.txt', To_write, fmt='%s')

    def run_prediction_on_unseen_with_y(self, train_main, test_main, data_loader, name):
        classes = np.load(test_main.main_param_path + '/classes.npy')
        if len(test_main.model_path) > 1:
            print("Do you want to predict using ensemble model ? If so then set the ensemble parameter to 1 and run "
                  "again")
        else:
            checkpoint_path = test_main.model_path[0]
            PATH = checkpoint_path + '/trained_model_' + name + '.pth'
            # PATH = data_loader.checkpoint_path + '/trained_model_' + name + '.pth'
            im_names = data_loader.filenames

            # print('im_names : {}'.format(im_names))
            checkpoint = torch.load(PATH, map_location='cpu')

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # device = torch.device("cpu")
            # self.model = self.model.module.to(device)

            avg_acc1, target, output, prob = cls_predict_on_unseen_with_y(train_main, test_main, data_loader.test_dataloader,
                                                                          self.model,
                                                                          self.criterion,
                                                                          time_begin=time())

            target = torch.cat(target)
            output = torch.cat(output)
            prob = torch.cat(prob)

            target = target.cpu().numpy()
            output = output.cpu().numpy()
            prob = prob.cpu().numpy()

            output_max = output.argmax(axis=1)

            target_label = np.array([classes[target[i]] for i in range(len(target))], dtype=object)
            output_label = np.array([classes[output_max[i]] for i in range(len(output_max))], dtype=object)

            output_corrected_label = copy.deepcopy(output_label)

            first_indices = prob.argsort()[:, -1]
            confs = [prob[i][first_indices[i]] for i in range(len(first_indices))]
            np.savetxt(test_main.test_outpath + '/Prediction_confidence.csv', confs)
            for i in range(len(confs)):
                if confs[i] < test_main.threshold:
                    output_corrected_label[i] = 'unknown'

            GT_Pred_GTLabel_PredLabel_Prob_PredLabelCorrected = [target, output_max, target_label, output_label,
                                                                 prob, output_corrected_label]
            with open(
                    test_main.test_outpath + '/Single_GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob_' + name
                    + '.pickle', 'wb') as cw:
                pickle.dump(GT_Pred_GTLabel_PredLabel_Prob_PredLabelCorrected, cw)

            output_label = output_label.tolist()

            if test_main.threshold > 0:
                print('I am using threshold value as : {}'.format(test_main.threshold))
                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_label)]
                np.savetxt(test_main.test_outpath + '/Single_model_Plankiformer_predictions.txt', To_write,
                           fmt='%s')

                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_corrected_label)]
                np.savetxt(test_main.test_outpath + '/Single_model_Plankiformer_predictions_thresholded.txt',
                           To_write, fmt='%s')

            else:
                print('I am using default value as threshold i.e. 0')
                To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], output_label)]
                np.savetxt(test_main.test_outpath + '/Single_model_Plankiformer_predictions.txt', To_write,
                           fmt='%s')

            accuracy_model = accuracy_score(target_label, output_label)
            clf_report = classification_report(target_label, output_label)
            clf_report_rm_0 = classification_report(target_label, output_label, labels=np.unique(target_label))
            f1 = f1_score(target_label, output_label, average='macro')
            f1_rm_0 = f1_score(target_label, output_label, average='macro', labels=np.unique(target_label))

            f = open(test_main.test_outpath + 'Single_test_report_' + name + '.txt', 'w')
            f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                                  clf_report))
            f.close()

            ff = open(test_main.test_outpath + 'Single_test_report_rm_0_' + name + '.txt', 'w')
            ff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1_rm_0,
                                                                                                  clf_report_rm_0))
            ff.close()

            ID_result = pd.read_pickle(test_main.model_path[0] + '/GT_Pred_GTLabel_PredLabel_prob_model_' + name + '.pickle')
            bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk, df_count, BC_AC, BC_PCC, BC_PAC = extra_metrics(target_label.tolist(), output_label, prob, ID_result)
            fff = open(test_main.test_outpath + 'Single_test_report_extra_' + name + '.txt', 'w')
            fff.write('\nbias\n\n{}\n\nBC\n\n{}\n\nMAE\n\n{}\n\nMSE\n\n{}\n\nRMSE\n\n{}\n\nR2\n\n{}\n\nNMAE\n\n{}\n\nAE_rm_junk\n\n{}\n\nNAE_rm_junk\n\n{}\n'.format(bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk))
            fff.close()

            df_count.to_excel(test_main.test_outpath + 'Population_count.xlsx', index=True, header=True)

            # CC, AC, PCC = quantification(target_label.tolist(), output_label, prob)

            plt.figure(figsize=(8, 6))
            plt.subplot(1, 1, 1)
            plt.xlabel('Class', fontsize=20)
            plt.ylabel('Count', fontsize=20)
            width = 0.5
            x = np.arange(0, len(df_count) * 2, 2)
            x1 = x - width / 2
            x2 = x + width / 2
            plt.bar(x1, df_count['Ground_truth'], width=0.5, label='Ground_truth')
            plt.bar(x2, df_count['Predict'], width=0.5, label='Prediction')
            plt.xticks(x, df_count.index, rotation=45, rotation_mode='anchor', ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(test_main.test_outpath + 'Population_count.png', dpi=300)
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.subplot(1, 1, 1)
            plt.xlabel('Class')
            plt.ylabel('Symmetric Absolute Percentage Error')
            x = range(len(df_count))
            plt.bar(x, abs(np.divide(df_count['Bias'], df_count['Ground_truth'] + df_count['Predict'])), width=0.5, label='SAPE')
            plt.xticks(x, df_count.index, rotation=45, rotation_mode='anchor', ha='right')
            # plt.legend()
            plt.tight_layout()
            plt.savefig(test_main.test_outpath + 'Population_Symmetric_Absolute_Percentage_Error.png', dpi=300)
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.subplot(1, 1, 1)
            plt.xlabel('Class')
            plt.ylabel('Normalized Absolute Error')
            x = range(len(df_count))
            plt.bar(x, abs(np.divide(df_count['Bias'], np.sum(df_count['Ground_truth']))), width=0.5, label='NAE')
            plt.xticks(x, df_count.index, rotation=45, rotation_mode='anchor', ha='right')
            # plt.legend()
            plt.tight_layout()
            plt.savefig(test_main.test_outpath + 'Population_Normalized_Absolute_Error.png', dpi=300)
            plt.close()


            plt.figure(figsize=(8, 6))
            plt.subplot(1, 1, 1)
            plt.xlabel('Class')
            plt.ylabel('Absolute Percentage Error')
            x = range(len(df_count.drop(df_count[df_count['Ground_truth'] == 0].index)))
            plt.bar(x, abs(np.divide(df_count.drop(df_count[df_count['Ground_truth'] == 0].index)['Bias'], df_count.drop(df_count[df_count['Ground_truth'] == 0].index)['Ground_truth'])), width=0.5, label='APE')
            plt.xticks(x, df_count.drop(df_count[df_count['Ground_truth'] == 0].index).index, rotation=45, rotation_mode='anchor', ha='right')
            # plt.legend()
            plt.tight_layout()
            plt.savefig(test_main.test_outpath + 'Population_Absolute_Percentage_Error.png', dpi=300)
            plt.close()

            labels = np.unique(target_label)
            unknown_index = np.where(labels=='unknown')[0][0]
            labels_rm_unknown = np.delete(labels, unknown_index)

            df_labels = pd.DataFrame(data=[target_label, output_label, prob])
            df_labels_rm_unknown = df_labels.drop(columns=df_labels.columns[df_labels.iloc[0] == 'unknown'])

            accuracy_rm_unknown = accuracy_score(df_labels_rm_unknown.iloc[0].tolist(), df_labels_rm_unknown.iloc[1].tolist())
            clf_report_rm_unknown = classification_report(target_label, output_label, labels=labels_rm_unknown)
            f1_rm_unknown = f1_score(target_label, output_label, average='macro', labels=labels_rm_unknown)

            ffff = open(test_main.test_outpath + 'Single_test_report_rm_unknown_' + name + '.txt', 'w')
            ffff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_rm_unknown, f1_rm_unknown,
                                                                                                  clf_report_rm_unknown))
            ffff.close()

            # bias_rm_unknown, BC_rm_unknown, MAE_rm_unknown, MSE_rm_unknown, RMSE_rm_unknown, R2_rm_unknown, weighted_recall_rm_unknown, df_count_rm_unknown = extra_metrics(df_labels_rm_unknown.iloc[0].tolist(), df_labels_rm_unknown.iloc[1].tolist(), df_labels_rm_unknown.iloc[2].tolist())
            # fffff = open(test_main.test_outpath + 'Single_test_report_extra_rm_unknown_' + name + '.txt', 'w')
            # fffff.write('\nbias\n\n{}\n\nBC\n\n{}\n\nMAE\n\n{}\n\nMSE\n\n{}\n\nRMSE\n\n{}\n\nR2\n\n{}\n\nweighted_recall\n\n{}\n'.format(bias_rm_unknown, BC_rm_unknown, MAE_rm_unknown, MSE_rm_unknown, RMSE_rm_unknown, R2_rm_unknown, weighted_recall_rm_unknown))
            # fffff.close()

            class_target = np.unique(target_label).tolist()
            class_output = np.unique(output_label).tolist()
            zero_support = list(set(class_output)-set(class_target))
            df_labels_rm_0 = pd.DataFrame(data=[target_label, output_label, prob])
            df_labels_rm_0_t = df_labels_rm_0.transpose()

            for i in zero_support:
                df_labels_rm_0_t = df_labels_rm_0_t[df_labels_rm_0_t.iloc[:, 1] != i]
            df_labels_rm_0 = df_labels_rm_0_t.transpose()
            # bias_rm_0, BC_rm_0, MAE_rm_0, MSE_rm_0, RMSE_rm_0, R2_rm_0, weighted_recall_rm_0, df_count_rm_0 = extra_metrics(df_labels_rm_0.iloc[0].tolist(), df_labels_rm_0.iloc[1].tolist(), df_labels_rm_0.iloc[2].tolist())
            # ffffff = open(test_main.test_outpath + 'Single_test_report_extra_rm_0_' + name + '.txt', 'w')
            # ffffff.write('\nbias\n\n{}\n\nBC\n\n{}\n\nMAE\n\n{}\n\nMSE\n\n{}\n\nRMSE\n\n{}\n\nR2\n\n{}\n\nweighted_recall\n\n{}\n'.format(bias_rm_0, BC_rm_0, MAE_rm_0, MSE_rm_0, RMSE_rm_0, R2_rm_0, weighted_recall_rm_0))
            # ffffff.close()

            filenames_out = im_names[0]
            filenames_out.reset_index(drop=True, inplace=True)
            for jj in range(len(filenames_out)):
                if target_label[jj] == output_label[jj]:
                    dest_path = test_main.test_outpath + '/' + name + '/Classified/' + str(target_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

                else:
                    dest_path = test_main.test_outpath + '/' + name + '/Misclassified/' + str(
                        target_label[jj]) + '_as_' + str(output_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

    def run_ensemble_prediction_on_unseen_with_y(self, train_main, test_main, data_loader, name):
        classes = np.load(test_main.main_param_path + '/classes.npy')
        Ensemble_prob = []
        Ensemble_GT = []
        Ensemble_GT_label = []
        im_names = data_loader.filenames

        for i in range(len(test_main.model_path)):
            checkpoint_path = test_main.model_path[i]
            PATH = checkpoint_path + '/trained_model_' + name + '.pth'

            # if torch.cuda.is_available() and test_main.use_gpu == 'yes':
            #     checkpoint = torch.load(PATH)
            # else:
            if torch.cuda.is_available() and test_main.use_gpu == 'yes':
                checkpoint = torch.load(PATH, map_location="cuda:" + str(test_main.gpu_id))
            else:
                checkpoint = torch.load(PATH, map_location='cpu')

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # device = torch.device("cpu")
            # self.model = self.model.module.to(device)

            # output, prob = cls_predict_on_unseen(data_loader.test_dataloader, self.model)
            avg_acc1, target, output, prob = cls_predict_on_unseen_with_y(train_main, test_main, data_loader.test_dataloader, self.model,
                                                                          self.criterion,
                                                                          time_begin=time())

            target = torch.cat(target)
            output = torch.cat(output)
            prob = torch.cat(prob)

            prob = prob.cpu().numpy()
            output = output.cpu().numpy()
            target = target.cpu().numpy()

            output_max = output.argmax(axis=1)

            target_label = np.array([classes[target[i]] for i in range(len(target))], dtype=object)
            output_label = np.array([classes[output_max[i]] for i in range(len(output_max))], dtype=object)

            GT_Pred_GTLabel_PredLabel_Prob = [target, output_max, target_label, output_label, prob]
            with open(
                    test_main.test_outpath + '/GT_Pred_GTLabel_PredLabel_Prob_' + name + '_' + str(i+1) +
                    '.pickle', 'wb') as cw:
                pickle.dump(GT_Pred_GTLabel_PredLabel_Prob, cw)

            Ensemble_prob.append(prob)
            Ensemble_GT.append(target)
            Ensemble_GT_label.append(target_label)

        Ens_DEIT_prob_max = []
        Ens_DEIT_label = []
        Ens_DEIT = []
        name2 = []
        GT_label = Ensemble_GT_label[0]
        GT = Ensemble_GT[0]

        if test_main.ensemble == 1:
            Ens_DEIT = sum(Ensemble_prob) / len(Ensemble_prob)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name2 = 'arth_mean_'

        elif test_main.ensemble == 2:
            Ens_DEIT = gmean(Ensemble_prob)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name2 = 'geo_mean_'

        Ens_DEIT_corrected_label = copy.deepcopy(Ens_DEIT_label)

        first_indices = Ens_DEIT.argsort()[:, -1]
        Ens_confs = [Ens_DEIT[i][first_indices[i]] for i in range(len(first_indices))]

        for i in range(len(Ens_confs)):
            if Ens_confs[i] < test_main.threshold:
                Ens_DEIT_corrected_label[i] = 'unknown'

        GT_Pred_GTLabel_PredLabel_Prob_PredLabelCorrected = [GT, Ens_DEIT_prob_max, GT_label, Ens_DEIT_label,
                                                             Ens_DEIT, Ens_DEIT_corrected_label]
        with open(
                test_main.test_outpath + '/Ensemble_GT_Pred_GTLabel_PredLabel_Prob_PredLabelCorrected_' + name2 + name +
                '.pickle', 'wb') as cw:
            pickle.dump(GT_Pred_GTLabel_PredLabel_Prob_PredLabelCorrected, cw)

        Ens_DEIT_label = Ens_DEIT_label.tolist()

        if test_main.threshold > 0:
            print('I am using threshold value as : {}'.format(test_main.threshold))

            ## Original
            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_label)]
            np.savetxt(test_main.test_outpath + '/Ensemble_models_predictions_' + name2 + name +
                       '.txt', To_write, fmt='%s')

            accuracy_model = accuracy_score(GT_label, Ens_DEIT_label)
            clf_report = classification_report(GT_label, Ens_DEIT_label)
            clf_report_rm_0 = classification_report(GT_label, Ens_DEIT_label, labels=np.unique(GT_label))
            f1 = f1_score(GT_label, Ens_DEIT_label, average='macro')
            f1_rm_0 = f1_score(GT_label, Ens_DEIT_label, average='macro', labels=np.unique(GT_label))

            f = open(test_main.test_outpath + 'Ensemble_test_report_' + name2 + name + '.txt', 'w')
            f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                                  clf_report))
            f.close()

            ff = open(test_main.test_outpath + 'Ensemble_test_report_rm_0_' + name2 + name + '.txt', 'w')
            ff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1_rm_0,
                                                                                                  clf_report_rm_0))
            ff.close()

            filenames_out = im_names[0]
            filenames_out.reset_index(drop=True, inplace=True)
            for jj in range(len(filenames_out)):
                if GT_label[jj] == Ens_DEIT_label[jj]:
                    dest_path = test_main.test_outpath + '/' + name2 + name + '/Classified/' + str(GT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)
                else:
                    dest_path = test_main.test_outpath + '/' + name2 + name + '/Misclassified/' + str(
                        GT_label[jj]) + '_as_' + str(Ens_DEIT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

            ## Thresholded

            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_corrected_label)]
            np.savetxt(test_main.test_outpath + '/Ensemble_models_predictions_' + name2 + name +
                       '_thresholded_' + str(test_main.threshold) + '.txt', To_write, fmt='%s')

            accuracy_model = accuracy_score(GT_label, Ens_DEIT_corrected_label)
            clf_report = classification_report(GT_label, Ens_DEIT_corrected_label)
            clf_report_rm_0 = classification_report(GT_label, Ens_DEIT_corrected_label, labels=np.unique(GT_label))
            f1 = f1_score(GT_label, Ens_DEIT_corrected_label, average='macro')
            f1_rm_0 = f1_score(GT_label, Ens_DEIT_corrected_label, average='macro', labels=np.unique(GT_label))

            f = open(test_main.test_outpath + 'Ensemble_test_report_' + name2 + name + '_thresholded_' + str(
                test_main.threshold) + '.txt', 'w')
            f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                                  clf_report))
            f.close()

            ff = open(test_main.test_outpath + 'Ensemble_test_report_rm_0_' + name2 + name + '_thresholded_' + str(
                test_main.threshold) + '.txt', 'w')
            ff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1_rm_0,
                                                                                                  clf_report_rm_0))
            ff.close()

            filenames_out = im_names[0]
            filenames_out.reset_index(drop=True, inplace=True)
            for jj in range(len(filenames_out)):
                if GT_label[jj] == Ens_DEIT_corrected_label[jj]:
                    dest_path = test_main.test_outpath + '/' + name2 + name + '_thresholded_' + str(
                        test_main.threshold) + '/Classified/' + str(GT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

                else:
                    dest_path = test_main.test_outpath + '/' + name2 + name + '_thresholded_' + str(
                        test_main.threshold) + '/Misclassified/' + str(
                        GT_label[jj]) + '_as_' + str(Ens_DEIT_corrected_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

        else:
            print('I am using default value as threshold i.e. 0')
            To_write = [i + '------------------ ' + j + '\n' for i, j in zip(im_names[0], Ens_DEIT_label)]
            np.savetxt(test_main.test_outpath + '/Ensemble_models_predictions_' + name2 + name +
                       '.txt', To_write, fmt='%s')

            accuracy_model = accuracy_score(GT_label, Ens_DEIT_label)
            clf_report = classification_report(GT_label, Ens_DEIT_label)
            clf_report_rm_0 = classification_report(GT_label, Ens_DEIT_label, labels=np.unique(GT_label))
            f1 = f1_score(GT_label, Ens_DEIT_label, average='macro')
            f1_rm_0 = f1_score(GT_label, Ens_DEIT_label, average='macro', labels=np.unique(GT_label))

            f = open(test_main.test_outpath + 'Ensemble_test_report_' + name2 + name + '.txt', 'w')
            f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                                  clf_report))
            f.close()

            ff = open(test_main.test_outpath + 'Ensemble_test_report_rm_0_' + name2 + name + '.txt', 'w')
            ff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1_rm_0,
                                                                                                  clf_report_rm_0))
            ff.close()

            ID_result = pd.read_pickle(test_main.model_path[0] + '/GT_Pred_GTLabel_PredLabel_prob_model_' + name + '.pickle')
            bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk, df_count, BC_AC, BC_PCC, BC_PAC = extra_metrics(GT_label.tolist(), Ens_DEIT_label, Ens_DEIT, ID_result)
            fff = open(test_main.test_outpath + 'Ensemble_test_report_extra_' + name2 + name + '.txt', 'w')
            fff.write('\nbias\n\n{}\n\nBC\n\n{}\n\nMAE\n\n{}\n\nMSE\n\n{}\n\nRMSE\n\n{}\n\nR2\n\n{}\n\nNMAE\n\n{}\n\nAE_rm_junk\n\n{}\n\nNAE_rm_junk\n\n{}\n'.format(bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk))
            fff.close()

            df_count.to_excel(test_main.test_outpath + 'Population_count.xlsx', index=True, header=True)

            # CC, AC, PCC = quantification(GT_label.tolist(), Ens_DEIT_label, Ens_DEIT)

            plt.figure(figsize=(8, 6))
            plt.subplot(1, 1, 1)
            plt.xlabel('Class', fontsize=20)
            plt.ylabel('Count', fontsize=20)
            width = 0.5
            x = np.arange(0, len(df_count) * 2, 2)
            x1 = x - width / 2
            x2 = x + width / 2
            plt.bar(x1, df_count['Ground_truth'], width=0.5, label='Ground_truth')
            plt.bar(x2, df_count['Predict'], width=0.5, label='Prediction')
            plt.xticks(x, df_count.index, rotation=45, rotation_mode='anchor', ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(test_main.test_outpath + 'Population_count.png', dpi=300)
            plt.close()

            labels = np.unique(GT_label)
            unknown_index = np.where(labels=='unknown')[0][0]
            labels_rm_unknown = np.delete(labels, unknown_index)

            df_labels = pd.DataFrame(data=[GT_label, Ens_DEIT_label, Ens_DEIT])
            df_labels_rm_unknown = df_labels.drop(columns=df_labels.columns[df_labels.iloc[0] == 'unknown'])

            # # for phyto
            # labels = np.unique(GT_label)
            # unknown_index = np.where(labels=='unknown')[0][0]
            # unknown_eccentric_index = np.where(labels=='unknown_eccentric')[0][0]
            # unknown_elongated_index = np.where(labels=='unknown_elongated')[0][0]
            # unknown_probably_dirt_index = np.where(labels=='unknown_probably_dirt')[0][0]
            # unrecognizable_dots_index = np.where(labels=='unrecognizable_dots')[0][0]
            # zooplankton_index = np.where(labels=='zooplankton')[0][0]

            # labels_rm_unknown = np.delete(labels, [unknown_index, unknown_eccentric_index, unknown_elongated_index, unknown_probably_dirt_index, unrecognizable_dots_index, zooplankton_index])

            # df_labels = pd.DataFrame(data=[GT_label, Ens_DEIT_label])
            # df_labels_rm_unknown = df_labels.drop(columns=df_labels.columns[df_labels.iloc[0] == 'unknown'])
            # df_labels_rm_unknown = df_labels_rm_unknown.drop(columns=df_labels.columns[df_labels.iloc[0] == 'unknown_eccentric'])
            # df_labels_rm_unknown = df_labels_rm_unknown.drop(columns=df_labels.columns[df_labels.iloc[0] == 'unknown_elongated'])
            # df_labels_rm_unknown = df_labels_rm_unknown.drop(columns=df_labels.columns[df_labels.iloc[0] == 'unknown_probably_dirt'])
            # df_labels_rm_unknown = df_labels_rm_unknown.drop(columns=df_labels.columns[df_labels.iloc[0] == 'unrecognizable_dots'])
            # df_labels_rm_unknown = df_labels_rm_unknown.drop(columns=df_labels.columns[df_labels.iloc[0] == 'zooplankton'])


            accuracy_rm_unknown = accuracy_score(df_labels_rm_unknown.iloc[0].tolist(), df_labels_rm_unknown.iloc[1].tolist())
            clf_report_rm_unknown = classification_report(GT_label, Ens_DEIT_label, labels=labels_rm_unknown)
            f1_rm_unknown = f1_score(GT_label, Ens_DEIT_label, average='macro', labels=labels_rm_unknown)

            ffff = open(test_main.test_outpath + 'Ensemble_test_report_rm_unknown_' + name2 + name + '.txt', 'w')
            ffff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_rm_unknown, f1_rm_unknown,
                                                                                                  clf_report_rm_unknown))
            ffff.close()

            # bias_rm_unknown, BC_rm_unknown, MAE_rm_unknown, MSE_rm_unknown, RMSE_rm_unknown, R2_rm_unknown, weighted_recall_rm_unknown, df_count_rm_unknown = extra_metrics(df_labels_rm_unknown.iloc[0].tolist(), df_labels_rm_unknown.iloc[1].tolist(), df_labels_rm_unknown.iloc[2].tolist())
            # fffff = open(test_main.test_outpath + 'Ensemble_test_report_extra_rm_unknown_' + name + '.txt', 'w')
            # fffff.write('\nbias\n\n{}\n\nBC\n\n{}\n\nMAE\n\n{}\n\nMSE\n\n{}\n\nRMSE\n\n{}\n\nR2\n\n{}\n\nweighted_recall\n\n{}\n'.format(bias_rm_unknown, BC_rm_unknown, MAE_rm_unknown, MSE_rm_unknown, RMSE_rm_unknown, R2_rm_unknown, weighted_recall_rm_unknown))
            # fffff.close()

            class_target = np.unique(target_label).tolist()
            class_output = np.unique(output_label).tolist()
            zero_support = list(set(class_output)-set(class_target))
            df_labels_rm_0 = pd.DataFrame(data=[target_label, output_label, prob])
            for i in zero_support:
                df_labels_rm_0 = df_labels_rm_0.drop(columns=df_labels_rm_0.columns[df_labels.iloc[0] == i])

            # bias_rm_0, BC_rm_0, MAE_rm_0, MSE_rm_0, RMSE_rm_0, R2_rm_0, weighted_recall_rm_0, df_count_rm_0 = extra_metrics(df_labels_rm_0.iloc[0].tolist(), df_labels_rm_0.iloc[1].tolist(), df_labels_rm_0.iloc[2].tolist())
            # ffffff = open(test_main.test_outpath + 'Ensemble_test_report_extra_rm_0_' + name + '.txt', 'w')
            # ffffff.write('\nbias\n\n{}\n\nBC\n\n{}\n\nMAE\n\n{}\n\nMSE\n\n{}\n\nRMSE\n\n{}\n\nR2\n\n{}\n\nweighted_recall\n\n{}\n'.format(bias_rm_0, BC_rm_0, MAE_rm_0, MSE_rm_0, RMSE_rm_0, R2_rm_0, weighted_recall_rm_0))
            # ffffff.close()


            filenames_out = im_names[0]
            filenames_out.reset_index(drop=True, inplace=True)
            for jj in range(len(filenames_out)):
                if GT_label[jj] == Ens_DEIT_label[jj]:
                    dest_path = test_main.test_outpath + '/' + name2 + name + '/Classified/' + str(GT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

                else:
                    dest_path = test_main.test_outpath + '/' + name2 + name + '/Misclassified/' + str(
                        GT_label[jj]) + '_as_' + str(Ens_DEIT_label[jj])
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    shutil.copy(filenames_out[jj], dest_path)

    def initialize_model(self, train_main, test_main, data_loader, lr):

        if torch.cuda.is_available() and train_main.use_gpu == 'yes':
            if test_main is None:
                device = torch.device("cuda:" + str(train_main.gpu_id))
            else:
                device = torch.device("cuda:" + str(test_main.gpu_id))
        else:
            device = torch.device("cpu")

        self.model.to(device)
        if data_loader.class_weights_tensor is not None:
            self.criterion = nn.CrossEntropyLoss(data_loader.class_weights_tensor)
        elif test_main is None:
            class_weights_tensor = torch.load(train_main.outpath + '/class_weights_tensor.pt')
            self.criterion = nn.CrossEntropyLoss(class_weights_tensor)
        else:
            class_weights_tensor = torch.load(test_main.outpath + '/class_weights_tensor.pt')
            self.criterion = nn.CrossEntropyLoss(class_weights_tensor)

        if torch.cuda.is_available() and train_main.use_gpu == 'yes':
            if test_main is None:
                if torch.cuda.is_available():
                    torch.cuda.set_device(train_main.gpu_id)
                    self.model.cuda(train_main.gpu_id)
                    self.criterion = self.criterion.cuda(train_main.gpu_id)
            else:
                if torch.cuda.is_available():
                    torch.cuda.set_device(test_main.gpu_id)
                    self.model.cuda(test_main.gpu_id)
                    self.criterion = self.criterion.cuda(test_main.gpu_id)

        # Observe that all parameters are being optimized
        # if train_main.last_layer_finetune == 'yes':
        #     self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                                        lr=train_main.lr, weight_decay=train_main.weight_decay)
        # else:
        #     self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_main.lr,
        #                                        weight_decay=train_main.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                               weight_decay=train_main.weight_decay)

    def load_model_and_run_prediction(self, train_main, test_main, data_loader):

        self.import_deit_models_for_testing(train_main, test_main,testing=True)

        if test_main.finetuned == 0:
            self.initialize_model(train_main, test_main, data_loader, train_main.lr)
            if test_main.ensemble == 0:
                self.run_prediction_on_unseen(train_main, test_main, data_loader, 'original')
            else:
                self.run_ensemble_prediction_on_unseen(train_main, test_main, data_loader, 'original')

        elif test_main.finetuned == 1:
            self.initialize_model(train_main, test_main, data_loader, train_main.lr)
            if test_main.ensemble == 0:
                self.run_prediction_on_unseen(train_main, test_main, data_loader, 'tuned')
            else:
                self.run_ensemble_prediction_on_unseen(train_main, test_main, data_loader, 'tuned')

        elif test_main.finetuned == 2:
            self.initialize_model(train_main, test_main, data_loader, train_main.lr)
            if test_main.ensemble == 0:
                self.run_prediction_on_unseen(train_main, test_main, data_loader, 'finetuned')
            else:
                self.run_ensemble_prediction_on_unseen(train_main, test_main, data_loader, 'finetuned')
        else:
            print('Choose the correct finetune label')

    def load_model_and_run_prediction_with_y(self, train_main, test_main, data_loader):

        self.import_deit_models_for_testing(train_main, test_main)

        if test_main.finetuned == 0:
            self.initialize_model(train_main, test_main, data_loader, train_main.lr)
            if test_main.ensemble == 0:
                self.run_prediction_on_unseen_with_y(train_main, test_main, data_loader, 'original')
            else:
                self.run_ensemble_prediction_on_unseen_with_y(train_main, test_main, data_loader, 'original')

        elif test_main.finetuned == 1:
            self.initialize_model(train_main, test_main, data_loader, train_main.lr)
            if test_main.ensemble == 0:
                self.run_prediction_on_unseen_with_y(train_main, test_main, data_loader, 'tuned')
            else:
                self.run_ensemble_prediction_on_unseen_with_y(train_main, test_main, data_loader, 'tuned')

        elif test_main.finetuned == 2:
            self.initialize_model(train_main, test_main, data_loader, train_main.lr)
            if test_main.ensemble == 0:
                self.run_prediction_on_unseen_with_y(train_main, test_main, data_loader, 'finetuned')
            else:
                self.run_ensemble_prediction_on_unseen_with_y(train_main, test_main, data_loader, 'finetuned')
        else:
            print('Choose the correct finetune label')


def adjust_learning_rate(optimizer, epoch, lr, warmup, disable_cos, epochs):
    lr = lr
    if epoch < warmup:
        lr = lr / (warmup - epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif not disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr


def show_images(data, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    print(data[1])
    ax.imshow(make_grid((data[0].detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


# class LRScheduler:
#     """
#     Learning rate scheduler. If the validation loss does not decrease for the
#     given number of `patience` epochs, then the learning rate will decrease by
#     by given `factor`.
#     """
#
#     def __init__(self, optimizer):
#         """
#         new_lr = old_lr * factor
#         :param optimizer: the optimizer we are using
#         """
#         self.optimizer = optimizer
#         # self.patience = patience
#         # self.min_lr = min_lr
#         # self.factor = factor
#         self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=5e-10)
#
#     def __call__(self):
#         self.lr_scheduler.step()


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(
            self, optimizer, patience=5, min_lr=1e-10, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            # mode='min',
            mode='max',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    # def __call__(self, val_loss):
    #     self.lr_scheduler.step(val_loss)
    def __call__(self, val_f1):
        self.lr_scheduler.step(val_f1)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def cls_train(train_main, train_loader, model, criterion, optimizer, clip_grad_norm, modeltype):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    targets = []
    outputs = []
    lr_scheduler = LRScheduler(optimizer)

    for i, (images, target) in enumerate(train_loader):

        if torch.cuda.is_available() and train_main.use_gpu == 'yes':
            device = torch.device("cuda:" + str(train_main.gpu_id))
        else:
            device = torch.device("cpu")

        images, target = images.to(device), target.to(device)

        if train_main.run_cnn_or_on_colab == 'yes':
            output = model(images)  # to run it on CSCS and colab
        else:
            output, x = model(images)

        if train_main.add_layer == 'yes':
            if train_main.architecture == 'deit' or train_main.architecture == 'vit' or train_main.architecture == 'mae':
                output = torch.mean(output, 1)

        loss = criterion(output, target.long())

        acc1 = accuracy(output, target)

        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()

        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm, norm_type=2)

        optimizer.step()

        outputs.append(output)
        targets.append(target)

    # if modeltype == 0:
    #     if train_main.run_lr_scheduler == 'yes':
    #         lr_scheduler(loss)

    outputs = torch.cat(outputs)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.argmax(outputs, axis=1)

    targets = torch.cat(targets)
    targets = targets.cpu().detach().numpy()

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    return avg_acc1, avg_loss, outputs, targets


def cls_validate(train_main, val_loader, model, criterion, time_begin=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    targets = []
    outputs = []

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available() and train_main.use_gpu == 'yes':
                device = torch.device("cuda:" + str(train_main.gpu_id))
            else:
                device = torch.device("cpu")

            images, target = images.to(device), target.to(device)

            output = model(images)

            if train_main.add_layer == 'yes':
                if train_main.architecture == 'deit' or train_main.architecture == 'vit' or train_main.architecture == 'mae':
                    output = torch.mean(output, 1)

            # loss = criterion(output, target)
            loss = criterion(output, target.long())
            acc1 = accuracy(output, target)

            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            outputs.append(output)
            targets.append(target)

        outputs = torch.cat(outputs)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.argmax(outputs, axis=1)

        targets = torch.cat(targets)
        targets = targets.cpu().detach().numpy()

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    return avg_acc1, avg_loss, outputs, targets, total_mins


def cls_predict(train_main, val_loader, model, criterion, time_begin=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    outputs = []
    targets = []
    probs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):

            if torch.cuda.is_available() and train_main.use_gpu == 'yes':
                device = torch.device("cuda:" + str(train_main.gpu_id))
            else:
                device = torch.device("cpu")

            images, target = images.to(device), target.to(device)
            targets.append(target)

            output = model(images)

            if train_main.add_layer == 'yes':
                if train_main.architecture == 'deit' or train_main.architecture == 'vit' or train_main.architecture == 'mae':
                    output = torch.mean(output, 1)

            outputs.append(output)
            prob = torch.nn.functional.softmax(output, dim=1)
            probs.append(prob)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    total_secs = -1 if time_begin is None else (time() - time_begin)
    print('Time taken for prediction (in secs): {}'.format(total_secs))

    return avg_acc1, targets, outputs, probs


def cls_predict_on_unseen(train_main, test_main, test_loader, model):
    model.eval()
    outputs = []
    probs = []
    time_begin = time()
    with torch.no_grad():
        for i, (images) in enumerate(test_loader):

            if torch.cuda.is_available() and test_main.use_gpu == 'yes':
                device = torch.device("cuda:" + str(test_main.gpu_id))
            else:
                device = torch.device("cpu")

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cuda")
            # images = torch.stack(images).to(device)
            images = images.to(device)

            output = model(images)

            if train_main.add_layer == 'yes':
                if train_main.architecture == 'deit' or train_main.architecture == 'vit' or train_main.architecture == 'mae':
                    output = torch.mean(output, 1)

            outputs.append(output)
            prob = torch.nn.functional.softmax(output, dim=1)
            probs.append(prob)

    total_secs = -1 if time_begin is None else (time() - time_begin)
    print('Time taken for prediction (in secs): {}'.format(total_secs))

    return outputs, probs


def cls_predict_on_unseen_with_y(train_main, test_main, val_loader, model, criterion, time_begin=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    outputs = []
    targets = []
    probs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available() and test_main.use_gpu == 'yes':
                device = torch.device("cuda:" + str(test_main.gpu_id))
            else:
                device = torch.device("cpu")

            images, target = images.to(device), target.to(device)
            targets.append(target)

            output = model(images)

            if train_main.add_layer == 'yes':
                if train_main.architecture == 'deit' or train_main.architecture == 'vit' or train_main.architecture == 'mae':
                    output = torch.mean(output, 1)

            outputs.append(output)
            prob = torch.nn.functional.softmax(output, dim=1)
            probs.append(prob)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    total_secs = -1 if time_begin is None else (time() - time_begin)
    print('Time taken for prediction (in secs): {}'.format(total_secs))

    return avg_acc1, targets, outputs, probs


def extra_metrics(GT_label, Pred_label, Pred_prob, ID_result):

    list_class = list(set(np.unique(GT_label)).union(set(np.unique(Pred_label))))
    list_class.sort()
    df_count_Pred_GT = pd.DataFrame(index=list_class, columns=['Predict', 'Ground_truth'])

    GT_label_ID = ID_result[2].tolist()
    Pred_label_ID = ID_result[3].tolist()
    Pred_prob_ID = ID_result[4]

    list_class_ID = np.unique(GT_label_ID).tolist()
    list_class_ID.sort()
    df_prob = pd.DataFrame(index=list_class_ID, columns=['prob'])
    for i in range(len(list_class_ID)):
        df_prob.iloc[i] = np.sum(Pred_prob[:, i])

    df_prob_ID_all = pd.DataFrame(data=Pred_prob_ID, columns=list_class_ID)

    CC = []
    AC = []
    PCC = []
    PAC = []

    for iclass in list_class:
        df_count_Pred_GT.loc[iclass, 'Predict'] = Pred_label.count(iclass)
        df_count_Pred_GT.loc[iclass, 'Ground_truth'] = GT_label.count(iclass)

        class_CC = Pred_label.count(iclass)
        CC.append(class_CC)

        true_copy, pred_copy = GT_label_ID.copy(), Pred_label_ID.copy()
        for i in range(len(GT_label_ID)):
            if GT_label_ID[i] == iclass:
                true_copy[i] = 1
            else:
                true_copy[i] = 0
            if Pred_label_ID[i] == iclass:
                pred_copy[i] = 1
            else:
                pred_copy[i] = 0
        tn, fp, fn, tp = confusion_matrix(true_copy, pred_copy).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        class_AC = (class_CC - (fpr * len(Pred_label))) / (tpr - fpr)
        AC.append(class_AC)

        class_PCC = df_prob.loc[iclass, 'prob']
        PCC.append(class_PCC)

        df_prob_ID = pd.DataFrame()
        df_prob_ID['Pred_label'] = Pred_label_ID
        df_prob_ID['GT_label'] = GT_label_ID
        df_prob_ID['Pred_prob'] = df_prob_ID_all[iclass]
        tpr_prob = np.sum(df_prob_ID[(df_prob_ID['GT_label'] == iclass) & (df_prob_ID['Pred_label'] == iclass)]['Pred_prob']) / (tp + fn)
        fpr_prob = np.sum(df_prob_ID[(df_prob_ID['GT_label'] != iclass) & (df_prob_ID['Pred_label'] == iclass)]['Pred_prob']) / (tn + fp)
        class_PAC = (class_PCC - (fpr_prob * len(Pred_label))) / (tpr_prob - fpr_prob)
        PAC.append(class_PAC)

    df_percentage_Pred_GT = df_count_Pred_GT.div(df_count_Pred_GT.sum(axis=0), axis=1)
    df_count_Pred_GT['Bias'] = df_count_Pred_GT['Predict'] - df_count_Pred_GT['Ground_truth']
    df_count_Pred_GT['CC'], df_count_Pred_GT['AC'], df_count_Pred_GT['PCC'], df_count_Pred_GT['PAC'] = CC, AC, PCC, PAC

    df_count_Pred_GT_rm_junk = df_count_Pred_GT.drop(['dirt', 'unknown', 'unknown_plankton'], errors='ignore')
    df_count_Pred_GT_rm_junk = df_count_Pred_GT_rm_junk.drop(df_count_Pred_GT_rm_junk[df_count_Pred_GT_rm_junk['Ground_truth'] == 0].index)

    df_count_Pred_GT_rm_0 = df_count_Pred_GT.drop(df_count_Pred_GT[df_count_Pred_GT['Ground_truth'] == 0].index)

    bias = np.sum(df_count_Pred_GT['Predict'] - df_count_Pred_GT['Ground_truth']) / df_count_Pred_GT.shape[0]
    BC = np.sum(np.abs(df_count_Pred_GT['Predict'] - df_count_Pred_GT['Ground_truth'])) / np.sum(np.abs(df_count_Pred_GT['Predict'] + df_count_Pred_GT['Ground_truth']))

    # Adjusted BC
    BC_AC = np.sum(np.abs(df_count_Pred_GT['AC'] - df_count_Pred_GT['Ground_truth'])) / np.sum(np.abs(df_count_Pred_GT['AC'] + df_count_Pred_GT['Ground_truth']))
    BC_PCC = np.sum(np.abs(df_count_Pred_GT['PCC'] - df_count_Pred_GT['Ground_truth'])) / np.sum(np.abs(df_count_Pred_GT['PCC'] + df_count_Pred_GT['Ground_truth']))
    BC_PAC = np.sum(np.abs(df_count_Pred_GT['PAC'] - df_count_Pred_GT['Ground_truth'])) / np.sum(np.abs(df_count_Pred_GT['PAC'] + df_count_Pred_GT['Ground_truth']))

    MAE = mean_absolute_error(df_count_Pred_GT['Ground_truth'], df_count_Pred_GT['Predict'])
    MSE = mean_squared_error(df_count_Pred_GT['Ground_truth'], df_count_Pred_GT['Predict'])
    RMSE = np.sqrt(MSE)
    R2 = r2_score(df_count_Pred_GT['Ground_truth'], df_count_Pred_GT['Predict'])

    AE_rm_junk = np.sum(np.abs(df_count_Pred_GT_rm_junk['Predict'] - df_count_Pred_GT_rm_junk['Ground_truth']))
    NAE_rm_junk = np.sum(np.divide(np.abs(df_count_Pred_GT_rm_junk['Predict'] - df_count_Pred_GT_rm_junk['Ground_truth']), df_count_Pred_GT_rm_junk['Ground_truth']))
    NMAE = np.mean(np.divide(np.abs(df_count_Pred_GT_rm_0['Predict'] - df_count_Pred_GT_rm_0['Ground_truth']), df_count_Pred_GT_rm_0['Ground_truth']))

    return bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk, df_count_Pred_GT, BC_AC, BC_PCC, BC_PAC
