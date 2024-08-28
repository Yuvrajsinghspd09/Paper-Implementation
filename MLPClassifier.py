'''
Develop MLPClassifier:
   * Wrap MLP with training and evaluation functionality
   * Implement data normalization and class weighting
   * Set up training loop with specified optimizer and loss function
'''



'''
Contents of config:
The config dictionary typically contains:

'input_size': Size of input features
'hidden_sizes': List of hidden layer sizes
'output_size': Size of output layer
'learning_rate': Learning rate for optimizer
'batch_size': Number of samples per batch
'epochs': Number of training epochs

Example config Dictionary
config = {
    'input_size': 768,
    'hidden_sizes': [512, 256],
    'output_size': 1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10
}
'''

class MLPClassifier:
  def __init__(self,config):
    self.config = config
    self.model = MLP(config['input_size'],config['hidden_sizes'],config['output_size'])
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config['learning_rate']
    self.criterion = nn.BCEWithLogitsLoss()


  def train(self,features,labels):
    #Sets model to training mode
    self.model.train()
    #Creates a TensorDataset and DataLoader for batch processing
    dataset = TensorDataset(features,labels)
    dataloader = DataLoader(dataset,batch_size = self.config['batch_size'],shuffle=True)
    
    #part of train
    #Performs the training loop: forward pass, loss calculation, backpropagation, and parameter update
    for epoch in range(self.config['epochs']):
      for batch_features,batch_labels in dataloader :
        self.optimizer.zero_grad()
        outputs = self.model(batch_features)
        loss = self.criterion(outputs, batch_labels.unsqueeze(1))
        loss.backward()
        self.optimizer.step()  
 
  def predict(self,features):
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(features)
    return torch.sigmoid(outputs)

'''
Sets model to evaluation mode
Disables gradient calculation for efficiency
Runs the model on input features
Applies sigmoid to get probabilities
'''



'''
Code Flow:

Initialize classifier with config
Call train method with features and labels
Training loop processes data in batches for specified epochs
Call predict method to get predictions on new data


'''


class MLPClassifier:
  def __init__(self,config):
    self.config = config
    self.model = MLP(config['input_size'],config['hidden_sizes'],config['output_size']
    self.optimizer = nn.optim.Adam(self.model.parameters(),lr=config['learning_rate']
    self.criterion = nn.BCEWithLogitsLoss()

  def train(self,features,labels):
    self.model.train()
    dataset = TensorDataset(features,labels)
    dataloader = DataLoader(dataset,batch_size=self.config['batch_size'])

    for epoch in range(self.config['epochs']):
      self.optimizer.zero_grad()
      outputs= self.model(batch_features)
      loss= self.criterion(outputs,batch_labels=unsqueeze(1))
      loss.backward()
      self.optimizer.step()

  def predict(self,features):
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(features)
    return torch.sigmoid(outputs)
