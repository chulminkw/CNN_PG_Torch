class EarlyStopping:
    def __init__(self, monitor='val_loss', mode='min', early_patience=5, verbose=1):
        self.monitor = monitor
        self.mode = mode
        self.early_patience = early_patience
        self.verbose = verbose
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0

    def is_improvement(self, value):
        if self.mode == 'min':
            return value < self.best_value
        else:
            return value > self.best_value

    def check_early_stop(self, value):
        is_early_stopped = False
        
        if self.is_improvement(value):
            self.best_value = value
            self.counter = 0
            is_early_stopped =False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.early_patience}")
            if self.counter >= self.early_patience:
                is_early_stopped = True
                if self.verbose:
                    print("Early stopping happens and train stops")
        
        return is_early_stopped
        