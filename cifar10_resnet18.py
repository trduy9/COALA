import coala  
import argparse  
import torch  
import torch.nn as nn  
from torchvision.models import resnet18, ResNet18_Weights  
from coala.models import BaseModel  

# Parse command line arguments  
parser = argparse.ArgumentParser(description='COALA CIFAR10 with Pretrained ResNet18 example')  
parser.add_argument('--num_clients', type=int, default=50, help='Number of clients')  
parser.add_argument('--participant_rate', type=float, default=0.1, help='Participant rate')  
parser.add_argument('--rounds', type=int, default=200, help='Number of rounds')  
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')  
parser.add_argument('--epochs', type=int, default=5, help='Local epochs')  
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Dirichlet distribution')  
parser.add_argument('--gpu', type=int, default=0, help='GPU to use (0 for CPU)')  
args = parser.parse_args()

# Device setup with GPU availability check
available_gpu_count = torch.cuda.device_count()
if available_gpu_count > 0 and args.gpu < available_gpu_count:
    device = torch.device(f"cuda:{args.gpu}")
    selected_gpu = args.gpu
else:
    print(f"[WARNING] Requested GPU {args.gpu} is not available. Switching to CPU.")
    device = torch.device("cpu")
    selected_gpu = 0

# Calculate clients per round from participant rate  
clients_per_round = max(1, int(args.num_clients * args.participant_rate))  

# Define a custom ResNet18 model that inherits from COALA's BaseModel  
class PretrainedResNet18(BaseModel):  
    def __init__(self, num_classes=10):  
        super(PretrainedResNet18, self).__init__()  
        # Load a pretrained ResNet18 model with weights from ImageNet  
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  

        # Replace the final fully connected layer to match CIFAR10 classes (10)  
        in_features = self.model.fc.in_features  
        self.model.fc = nn.Linear(in_features, num_classes)  

    def forward(self, x):  
        return self.model(x)  

# Register our custom model with COALA  
coala.register_model(PretrainedResNet18(num_classes=10))  

# Define COALA configuration  
config = {  
    "data": {  
        "dataset": "cifar10",  
        "num_of_clients": args.num_clients,  
        #"split_type": "dir",  
        "alpha": args.alpha,  
        "min_size": 10  
    },  
    "server": {  
        "rounds": args.rounds,  
        "clients_per_round": clients_per_round,  
        "test_every": 1,  
        "aggregation_strategy": "FedAvg"  
    },  
    "client": {  
        "batch_size": args.batch_size,  
        "test_batch_size": 32,
        "local_epoch": args.epochs,  
        "optimizer": {  
            "type": "SGD",  
            "lr": 0.01,  
            "momentum": 0.9,  
            "weight_decay": 0.0005 
        }  
    },  
    "test_mode": "test_in_server",  
    "gpu": selected_gpu  
}  

# Initialize COALA with our configuration  
coala.init(config)  

# Run federated learning  
coala.run()