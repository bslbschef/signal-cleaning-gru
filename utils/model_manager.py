import torch
import os


class ModelManager:
    def __init__(self, model, model_dir='models'):
        self.model = model
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def save_model(self, model_name='gru_model', epoch=None):
        """保存模型，支持版本化保存"""
        if epoch:
            model_file = os.path.join(self.model_dir, f"{model_name}_epoch_{epoch}.pth")
        else:
            model_file = os.path.join(self.model_dir, f"{model_name}_final.pth")

        torch.save(self.model.state_dict(), model_file)
        print(f"Model saved at {model_file}")

    def load_model(self, model_name='gru_model', epoch=None):
        """加载指定版本的模型"""
        if epoch:
            model_file = os.path.join(self.model_dir, f"{model_name}_epoch_{epoch}.pth")
        else:
            model_file = os.path.join(self.model_dir, f"{model_name}_final.pth")

        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file))
            print(f"Model loaded from {model_file}")
        else:
            print(f"Model file {model_file} not found.")
