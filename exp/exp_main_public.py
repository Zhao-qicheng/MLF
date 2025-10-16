
from tqdm.auto import tqdm

from exp.exp_basic import Exp_Basic
from models import MLF,PathFormer,PatchTST_ScaleFormer,PatchTST,NHits_Scaleformer,Autoformer_Scaleformer,NHits,FiLM

from utils.tools import  adjust_learning_rate, visual
from utils.metrics import metric,MAPE_Fund
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from data_provider.data_factory_pubilc import data_provider

import json
warnings.filterwarnings('ignore')

class moving_avg(nn.Module):
    def __init__(self):
        super(moving_avg, self).__init__()
    def forward(self, x, kernel_size):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            convert_numpy = True
            x = torch.tensor(x)
        else:
            convert_numpy = False
        x = nn.functional.avg_pool1d(x.permute(0, 2, 1), kernel_size, kernel_size)
        x = x.permute(0, 2, 1)
        if convert_numpy:
            x = x.numpy()
        return x

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.mv = moving_avg()
        self.configs=args
        self.criterion_rec = nn.MSELoss()
    def _build_model(self):

        model_dict = {

            'MLF':MLF,
            'PathFormer':PathFormer,
            'PatchTST_SFormer':PatchTST_ScaleFormer,
            'PatchTST':PatchTST,
            'NHits_Scaleformer': NHits_Scaleformer,
            'Autoformer_Scaleformer': Autoformer_Scaleformer,
            'NHits': NHits,
            'FiLM': FiLM,

        }
        model = model_dict[self.args.model].Model(configs=self.args).float()

        # model = MLF.Model(self.args).float()

        print(f"NUMBER OF PARAMETERS IN MODEL: {self.args.model}: {sum(p.numel() for p in model.parameters())}")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, additional_params=None):
        if additional_params is not None:
            model_optim = optim.AdamW(list(self.model.parameters())+additional_params, lr=self.args.learning_rate)
        else:
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion,mode=None):
        total_loss = []
        self.model.eval()
        preds=[]
        trues=[]
        flag=False
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.model == 'PathFormer':
                    outputs, balance_loss = self.model(batch_x)
                elif self.args.model == 'MLF':
                    outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, scale_all_rec, scale_all_patch = outputs_all
                elif self.args.model == 'PatchTST_SFormer':
                    outputs_all, _ = self.model(batch_x)
                    outputs = outputs_all[-1]
                elif self.args.model == 'PatchTST':
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)

        pred_all = np.array(np.concatenate(preds,axis=0))
        trues_all = np.array(np.concatenate(trues,axis=0))
        total_loss=self.criterion(pred=pred_all,true=trues_all)

        self.model.train()
        return total_loss

    def train_one_epoch(self,i,batch_x,batch_y,batch_x_mark,batch_y_mark,iter_count):
        loss=0
        epoch_time = time.time()
        iter_count += 1
        self.model_optim.zero_grad()
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.model == 'PathFormer':
            outputs, balance_loss = self.model(batch_x)
        elif self.args.model=='MLF':
            outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, scale_all_rec, scale_all_patch=outputs_all
        elif self.args.model=='PatchTST_SFormer':
            outputs_all,_ = self.model(batch_x)
            outputs = outputs_all[-1]
        elif  self.args.model=='PatchTST':
            outputs= self.model(batch_x)

        rec_loss = 0

        if 'MLF' in self.configs.model and self.configs.patch_squeeze and self.configs.reconstruct_loss and len(scale_all_rec.keys())!=0:
            for scale in scale_all_rec.keys():
                rec_loss += self.criterion_rec(scale_all_rec[scale], scale_all_patch[scale])
            rec_loss /= (len(scale_all_rec.keys()))

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        if 'PatchTST_SFormer' in self.args.model or 'MS' in self.args.model:
            for li, (scale, output) in enumerate(zip(self.args.scales[:-1], outputs_all[:-1])):
                tmp_y = self.mv(batch_y, scale)
                tmp_loss=self.criterion_tmp(output, tmp_y).mean()
                loss += tmp_loss / scale
            loss = loss / 2
        else:
            loss=self.criterion_tmp(outputs, batch_y).mean()
        if self.args.model == 'PathFormer':
            loss+=balance_loss
        elif self.args.model=='MLF':
            loss+=rec_loss
        loss.backward()
        self.model_optim.step()

        return loss.item()
    def save_checkpoint(self, val_loss, model, path):
        # if self.verbose:
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + self.args.script_id+'checkpoint.pth')
        self.val_loss_min = val_loss
    def train(self, setting):

        self.criterion =MAPE_Fund(self.args).cal_fund_val
        train_data, train_loader = self._get_data(flag='train')

        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self.test_loader=test_loader
        self.test_loader=test_loader

        self.criterion_tmp = torch.nn.MSELoss(reduction='none')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps=0
        self.model_optim = self._select_optimizer()
        train_loss_all_dict={}
        valid_loss_all_dict={}
        test_loss_all_dict={}
        time_now = time.time()
        epoch_time_all=[]
        self.val_loss_min=100
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.args.epoch = epoch
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                self.args.test = False
                loss=self.train_one_epoch(i,batch_x, batch_y, batch_x_mark, batch_y_mark,iter_count)
                train_loss.append(loss)
            epoch_time_all.append(time.time() - epoch_time)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_all_dict[epoch]=train_loss
            vali_data=None
            test_data=None
            self.args.mode='valid'
            self.args.test = True
            vali_loss_dict = self.vali(vali_data, vali_loader, self.criterion,mode='valid')
            valid_loss_all_dict[epoch]=vali_loss_dict
            vali_loss = vali_loss_dict['mse']
            self.args.mode = 'test'
            test_loss_dict = self.vali(test_data, self.test_loader, self.criterion,mode='test')
            test_loss_all_dict[epoch]=test_loss_dict
            test_loss = test_loss_dict['mse']

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if self.args.speed_mode:
                pass
            else:
                if vali_loss < self.val_loss_min:
                    self.val_loss_min = vali_loss
                    self.save_checkpoint(vali_loss, self.model, path)

            adjust_learning_rate(self.model_optim, epoch + 1, self.args)

            json_record_loss_train = json.dumps(train_loss_all_dict, indent=4)
            json_record_loss_val = json.dumps(valid_loss_all_dict, indent=4)
            json_record_loss_test = json.dumps(test_loss_all_dict, indent=4)
            if self.args.record:
                with open(path + '/record_all_loss_train' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_train)
                with open(path + '/record_all_loss_val' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_val)
                with open(path + '/record_all_loss_test' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_test)
        train_cost = time.time() - time_now
        train_loss_all_dict['train_cost_time'] = train_cost
        train_loss_all_dict['train_mean_epoch_time'] = np.mean(epoch_time_all)

        json_record_loss_train = json.dumps(train_loss_all_dict, indent=4)
        json_record_loss_val = json.dumps(valid_loss_all_dict, indent=4)
        json_record_loss_test = json.dumps(test_loss_all_dict, indent=4)

        if self.args.record:
            with open(path + '/record_all_loss_train' + '.json', 'w') as json_file:
                json_file.write(json_record_loss_train)
            with open(path + '/record_all_loss_val' + '.json', 'w') as json_file:
                json_file.write(json_record_loss_val)
            with open(path + '/record_all_loss_test' + '.json', 'w') as json_file:
                json_file.write(json_record_loss_test)

        # train_cost=time.time()-time_now
        # test_loss_all_dict['train_cost_time']=train_cost
        if not self.args.speed_mode:
            best_model_path = path + '/' + self.args.script_id+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    def test_efficiency(self, setting, test=0):
        self.criterion_tmp = torch.nn.MSELoss(reduction='none')
        self.args.device='cuda:'+str(self.args.gpu)
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.configs.path=path
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        self.test_loader = test_loader
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.configs.path=path
        # best_model_path = path + '/' + self.args.script_id + 'checkpoint.pth'
        self.criterion = MAPE_Fund(self.args).cal_fund_val
        self.args.mode = 'test'
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        running_times = []
        test_mse=[]
        time_now=time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark)in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                start_time = time.time()
                if self.args.model == 'PathFormer':
                    outputs, balance_loss = self.model(batch_x)
                elif self.args.model == 'MLF':
                    outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, scale_all_rec, scale_all_patch = outputs_all
                elif self.args.model == 'PatchTST_SFormer':
                    outputs_all, _ = self.model(batch_x)
                    outputs = outputs_all[-1]
                elif self.args.model == 'PatchTST':
                    outputs = self.model(batch_x)
                running_times.append(time.time()-start_time)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs.numpy()
                true = batch_y.numpy()
                preds.append(pred)
                trues.append(true)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark)in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                start_time = time.time()
                if self.args.model == 'PathFormer':
                    outputs, balance_loss = self.model(batch_x)
                elif self.args.model == 'MLF':
                    outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, scale_all_rec, scale_all_patch = outputs_all
                elif self.args.model == 'PatchTST_SFormer':
                    outputs_all, _ = self.model(batch_x)
                    outputs = outputs_all[-1]
                elif self.args.model == 'PatchTST':
                    outputs = self.model(batch_x)
                running_times.append(time.time()-start_time)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs.numpy()
                true = batch_y.numpy()
                preds.append(pred)
                trues.append(true)

        print('Inference time: ', time.time() - time_now)

        return
    def test(self, setting, test=0):
        if self.args.loss=='mse':
            self.criterion_tmp = torch.nn.MSELoss(reduction='none')
        elif self.args.loss=='huber':
            self.criterion_tmp = torch.nn.HuberLoss(reduction='none', delta=0.5)
        elif self.args.loss=='l1':
            self.criterion_tmp = torch.nn.L1Loss(reduction='none')
        self.args.device='cuda:'+str(self.args.gpu)
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.configs.path=path
        test_data, test_loader = self._get_data(flag='test')
        self.test_loader = test_loader
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.configs.path=path
        best_model_path = path + '/' + self.args.script_id + 'checkpoint.pth'
        self.criterion = MAPE_Fund(self.args).cal_fund_val
        
        '''
        test=1（重新加载权重）: 
        在评测开始前，从 checkpoints/<setting>/<script_id>checkpoint.pth 强制加载保存的最优模型再测试。
        适合单独跑测试、复现结果、或训练和测试分开在不同进程/时间执行的场景。
        '''
        '''
        test=0（不重新加载）: 
        直接用内存中当前的 self.model 做评测。
        若刚刚跑完 train()，两者通常等价，因为训练末尾已把最佳权重加载回模型。
        适合训练和测试在同一进程/时间执行的场景。
        '''
        self.args.mode = 'test'
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        running_times = []
        test_mse=[]
        time_now=time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark)in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                start_time = time.time()
                outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, scale_all_rec, scale_all_patch = outputs_all
                running_times.append(time.time()-start_time)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs.numpy()
                true = batch_y.numpy()
                preds.append(pred)
                trues.append(true)


        preds = np.array(np.concatenate(preds,axis=0))
        trues = np.array(np.concatenate(trues,axis=0))
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        # print(self.criterion(preds,trues))
        total_loss_dict=self.criterion(preds,trues)

        print('num test batch {} {} mean metric {}'.format(i,len(test_mse),np.mean(test_mse)))

        test_cost_time=time.time()-time_now
        total_loss_dict['test_time']=test_cost_time

        json_record_loss_test = json.dumps(total_loss_dict, indent=4)
        with open(path + '/final_test' + '.json', 'w') as json_file:
            json_file.write(json_record_loss_test)
        print('final test',total_loss_dict)

        return
