import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from modular.v1 import ModelCheckpoint, EarlyStopping


class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, scheduler=None, 
                 callbacks=None, device=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        # scheduler �߰�
        self.scheduler = scheduler
        self.device = device
        # ���� learning rate ���� �߰�
        self.current_lr = self.optimizer.param_groups[0]['lr']
        #checkpoint�� early stopping Ŭ�������� list�� ����. 
        self.callbacks = callbacks
        
    def train_epoch(self, epoch):
        self.model.train()

        # running ��� loss ���.
        accu_loss = 0.0
        running_avg_loss = 0.0
        # running ��� metric(��Ȯ��) ���. ���� runnig_avg_metric�� ��ü �н� �������� ��Ȯ��
        accu_metric = 0.0
        running_avg_metric = 0.0
        # epoch�� ��Ȯ�� ����� ���� ��ü �Ǽ� �� ���� ��Ȯ�Ǽ�
        num_total = 0.0
        accu_num_correct = 0.0
        # tqdm���� �ǽð� training loop ���� ��Ȳ �ð�ȭ
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1} [Training..]", leave=True) as progress_bar:
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                # �ݵ�� to(self.device). to(device) �ƴ�.
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # batch �ݺ� �� ���� ����  loss�� ���ϰ� �̸� batch Ƚ���� ������ running ��� loss ����.
                accu_loss += loss.item()
                running_avg_loss = accu_loss /(batch_idx + 1)

                # ��ġ�� accuracy metric ���
                num_correct = (outputs.argmax(-1) == targets).sum().item()
                metric_value = num_correct / inputs.shape[0] # num_correct / batch ũ��
                # running ��� accuracy metric ���. ��ġ�� ��Ȯ���� ��� ���� �ڿ� ��ġ Ƚ���� ����. �������δ� ��ü �������� ��Ȯ���� ����.
                accu_metric += metric_value
                accuracy = accu_metric / (batch_idx + 1)

                # epoch������ ��Ȯ�� ����� ���� ��ü �Ǽ��� ��ü num_correct �Ǽ� ���  
                num_total += inputs.shape[0]
                accu_num_correct += num_correct

                #tqdm progress_bar�� ���� ��Ȳ �� running ��� loss�� ��Ȯ�� ǥ��
                progress_bar.update(1)
                if batch_idx % 20 == 0 or (batch_idx + 1) == progress_bar.total:  # 20 batch Ƚ������ �Ǵ� �� ������ batch���� update
                    progress_bar.set_postfix({"Loss": running_avg_loss,
                                              "Accuracy": accuracy})

        if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
            self.current_lr = self.scheduler.get_last_lr()[0]
        
        accuracy = accu_num_correct / num_total
        return running_avg_loss, accuracy

    def validate_epoch(self, epoch):
        if not self.val_loader:
            return None

        self.model.eval()

        # running ��� loss ���.
        accu_loss = 0
        running_avg_loss = 0
        # running ��� metric(��Ȯ��) ���. ���� runnig_avg_metric�� ��ü �н� �������� ��Ȯ��
        accu_metric = 0.0
        running_avg_metric = 0.0
        # epoch�� ��Ȯ�� ����� ���� ��ü �Ǽ� �� ���� ��Ȯ�Ǽ�
        num_total = 0.0
        accu_num_correct = 0.0
        current_lr = self.optimizer.param_groups[0]['lr']
        with tqdm(total=len(self.val_loader), desc=f"Epoch {epoch+1} [Validating]", leave=True) as progress_bar:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)

                    loss = self.loss_fn(outputs, targets)
                    # batch �ݺ� �� ���� ����  loss�� ���ϰ� �̸� batch Ƚ���� ������ running ��� loss ����.
                    accu_loss += loss.item()
                    running_avg_loss = accu_loss /(batch_idx + 1)

                    # ��ġ�� accuracy metric ���
                    num_correct = (outputs.argmax(-1) == targets).sum().item()
                    metric_value = num_correct / inputs.shape[0] # num_correct / batch ũ��
                    # running ��� accuracy metric ���. ��ġ�� ��Ȯ���� ��� ���� �ڿ� ��ġ Ƚ���� ����. �������δ� ��ü �������� ��Ȯ���� ����.
                    accu_metric += metric_value
                    accuracy = accu_metric / (batch_idx + 1)
                    
                    # epoch������ ��Ȯ�� ����� ���� ��ü �Ǽ��� ��ü num_correct �Ǽ� ���  
                    num_total += inputs.shape[0]
                    accu_num_correct += num_correct
                    
                    #tqdm progress_bar�� ���� ��Ȳ �� running ��� loss�� ��Ȯ�� ǥ��
                    progress_bar.update(1)
                    if batch_idx % 20 == 0 or (batch_idx + 1) == progress_bar.total:  # 20 batch Ƚ������ �Ǵ� �� ������ batch���� update
                        progress_bar.set_postfix({"Loss": running_avg_loss,
                                                  "Accuracy":accuracy})
        # scheduler�� ���� ������ ��ݿ��� epoch������ ���� loss�� �Է�����.
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(running_avg_loss)
            self.current_lr = self.scheduler.get_last_lr()[0]

        accuracy = accu_num_correct / num_total
        return running_avg_loss, accuracy

    def fit(self, epochs):
        # epoch �ø��� �н�/���� ����� ����ϴ� history dict ����. learning rate �߰�
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f} Train Accuracy: {train_acc:.4f}",
                  f", Val Loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}" if val_loss is not None else "",
                  f", Current lr:{self.current_lr:.6f}")
            # epoch �ø��� �н�/���� ����� ���. learning rate �߰�
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            history['lr'].append(self.current_lr)
            
            if(self.callbacks):
                is_epoch_loop_break = self._execute_callbacks(self.callbacks, self.model, epoch, val_loss, val_acc)
                if is_epoch_loop_break:
                    break
                                
        return history

    def _execute_callbacks(self, callbacks, model, epoch, val_loss, val_acc):
        is_early_stopped = False
        
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if callback.monitor == 'val_loss':    
                    callback.save(model, epoch, val_loss)
                elif callback.monitor == 'val_acc':
                    callback.save(model, epoch, val_acc)
            if isinstance(callback, EarlyStopping):
                if callback.monitor == 'val_loss':
                    is_early_stopped = callback.check_early_stop(val_loss)
                if callback.monitor == 'val_acc':
                    is_early_stopped = callback.check_early_stop(val_acc)
                
        return is_early_stopped

    # �н��� �Ϸ�� ���� return
    def get_trained_model(self):
        return self.model