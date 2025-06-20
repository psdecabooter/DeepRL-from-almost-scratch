import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Q-network
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def save_policy(policy, path='policy.pth', additional_info=None):
    """
    Save the trained policy model
    
    Args:
        policy: Your trained PyTorch model
        path: Path to save the model
        additional_info: Optional dictionary containing extra information to save
                        (e.g., training hyperparameters, performance metrics)
    """
    save_dict = {
        'model_state_dict': policy.state_dict(),
    }
    
    if additional_info is not None:
        save_dict.update(additional_info)
    
    torch.save(save_dict, path)
    print(f"Policy saved to {path}")

def load_policy(path, policy_class=None, *args, device: str = 'cpu'):
    """
    Load a saved policy model
    
    Args:
        path: Path to the saved model
        policy_class: The policy class/architecture (needed to reconstruct the model)
        device: Device to load the model onto ('cpu' or 'cuda')
    
    Returns:
        The loaded policy and any additional saved information
    """
    checkpoint = torch.load(path, map_location=device)
    
    if policy_class is None:
        policy = None
        print("Warning: No policy class provided. Only state dict loaded.")
    else:
        policy = policy_class(*args).to(device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()  # Set to evaluation mode
    
    # Return everything that was saved
    additional_info = {k: v for k, v in checkpoint.items() 
                      if k != 'model_state_dict'}
    
    return policy, additional_info
