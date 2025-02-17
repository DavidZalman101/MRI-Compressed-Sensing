import torch
import os
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, mask_lr, results_root, drop_rate, learn_mask=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.mask_lr = mask_lr
        self.results_root = results_root
        self.early_stopping_patience = 5
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0
        self.best_model_state = None
        self.learn_mask = learn_mask
        self.drop_rate = drop_rate
        self.train_psnr_mean = 0
        self.train_psnr_std = 0
        self.test_psnr_mean = 0
        self.test_psnr_std = 0
        self.train_losses = []
        self.val_losses = []
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss, train_psnr, train_psnr_std = self.train_epoch(train_loader)
            val_loss, val_psnr, val_psnr_std = self.evaluate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f'Epopch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train PSNR: {train_psnr}, Val PSNR: {val_psnr} ')

            self.scheduler.step(val_loss)


            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1
                if self.no_improve_epochs >= self.early_stopping_patience:
                    print('Early stopping triggered.')
                    break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        self.train_psnr_mean, self.train_psnr_std = train_psnr, train_psnr_std
        self.plot_losses()


    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        psnr_values = []
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            if self.model.subsample.learn_mask:
                self.model.subsample.mask_grad(self.mask_lr)
            
            total_loss += loss.item()
            psnr_values.append(self.calculate_psnr(outputs, targets))

            del inputs, outputs, targets, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(loader)
        avg_psnr = np.mean(psnr_values)
        psnr_std = np.std(psnr_values)
        return avg_loss, avg_psnr, psnr_std

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        psnr_values = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                psnr_values.append(self.calculate_psnr(outputs, targets))
        
        avg_loss = total_loss / len(loader)
        avg_psnr = np.mean(psnr_values)
        psnr_std = np.std(psnr_values)
        return avg_loss, avg_psnr, psnr_std

    def calculate_psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel_value = torch.max(target) - torch.min(target)  # Updated data range calculation
        psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
        return psnr.item()

    def save_psnr_results(self):
            os.makedirs(f'{self.results_root}/psnr', exist_ok=True)
            mask_status = "learned_mask" if self.learn_mask else "unlearned_mask"
            with open(f'{self.results_root}/psnr/{mask_status}_{self.drop_rate}.txt', 'w') as f:
                f.write(f'Train PSNR mean: {self.train_psnr_mean}\n')
                f.write(f'Train PSNR std: {self.train_psnr_std}\n')
                f.write(f'Test PSNR mean: {self.test_psnr_mean}\n')
                f.write(f'Test PSNR std: {self.test_psnr_std}\n')

    def test(self, test_loader):
        _, self.test_psnr_mean, self.test_psnr_std = self.evaluate(test_loader)
        self.save_images(test_loader)

    def save_images(self, test_loader):
        self.model.eval()
        freq, image = next(iter(test_loader))
        output = self.model(freq.to(self.device)).squeeze(1)

        os.makedirs(f'{self.results_root}/images', exist_ok=True)
        mask_status = "learned_mask" if self.learn_mask else "unlearned_mask"
        
        # Save output image
        plt.imshow(output[0].detach().cpu().numpy(), cmap='gray')
        plt.savefig(f'{self.results_root}/images/{mask_status}_output_{self.drop_rate}.png')
        plt.close()
        
        # Save target image
        plt.imshow(image[0].detach().cpu().numpy(), cmap='gray')
        plt.savefig(f'{self.results_root}/images/{mask_status}_true_{self.drop_rate}.png')
        plt.close()

    def plot_losses(self, i = 0):
        os.makedirs(f'{self.results_root}/graphs', exist_ok=True)
        mask_status = "learned_mask" if self.learn_mask else "unlearned_mask"
        
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.results_root}/graphs/{mask_status}_loss_graph_{self.drop_rate}.png')
        plt.close()