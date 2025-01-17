import os
import torch

class ModelCheckpoint:
    def __init__(self, checkpoint_dir='checkpoints', monitor='val_loss', mode='min', save_interval=1, verbose=1):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.verbose = verbose
        self.save_interval = save_interval
        self._make_checkpoint_dir_unless()

    def _make_checkpoint_dir_unless(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    # mode 유형에 따라 metric value값이 이전 epoch시 보다 향상 되었는지 확인하여 True/False 값 return
    def is_improvement(self, value):
        if self.mode == 'min':
            return value < self.best_value
        else:
            return value > self.best_value

    # self.best_value값 update, is_improvement() 반환값이 True인 경우만 수행. 
    def update_best_value(self, value):
        self.best_value = value

    def save(self, model, epoch, value):
        if self.save_interval == 1:
            if self.is_improvement(value):
                self._checkpoint_save(model, epoch, value)
                self.update_best_value(value)
            
        elif self.save_interval > 1:
            if (epoch + 1) % self.save_interval == 0:
                self._checkpoint_save(model, epoch, value)
                 
        # 수행하지 말고 참조만 할 것(save_interval 횟수마다 model 성능이 향상되는 경우 저장)
        # if (epoch + 1) % self.save_interval == 0 and self.is_improvement(value):
        #     self.update_best_value(value)
        #     self._checkpoint_save(model, epoch, value)
            
    def _checkpoint_save(self, model, epoch, value):
        checkpoint_path = os.path.join(self.checkpoint_dir, 
                                       f'checkpoint_epoch_{epoch+1}_{self.monitor}_{value:.4f}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        if self.verbose:
            print(f"Saved model checkpoint at {checkpoint_path}")