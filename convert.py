from pytorch2keras import pytorch_to_keras
import torch
from torch import nn
from torch.autograd import Variable

### Pytorch model
class MyModel(nn.Module):
    """
    Module for Conv2d testing
    """

    def __init__(self):
        super().__init__()
        self.blstm_1 = nn.LSTM(257, 200, batch_first=True, bidirectional=True)
        self.blstm_2 = nn.LSTM(200, 200, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.blstm_1(x)
        x = self.blstm_2(x)
        return x

# de_model = Sequential()

# de_model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat', input_shape=(None, 257))) #dropout=0.15, recurrent_dropout=0.15
# de_model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat'))

# de_model.add(TimeDistributed(Dense(300)))
# de_model.add(LeakyReLU())
# de_model.add(Dropout(0.05))

# de_model.add(TimeDistributed(Dense(257)))
# de_model.add(Activation('sigmoid'))

model = MyModel()
# model.load_state_dict(torch.load("./models/best_netG.pt"))
torch.save(model.state_dict(), "tmp.pt")

# ### Dummy
# input_np = np.random.uniform(0, 1, (1, 16000, 257))
# input_var = Variable(torch.FloatTensor(input_np))

# ### Convert
# k_model = pytorch_to_keras(model, input_var, [(None, 257)], verbose=True)  
# k_model.save_weights("./models/best_netG.hdf5")