import torch
import torchvision.models as models
import classNN
model = classNN.NeuralNetwork() # models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
print(model)

model2 = torch.load('model.pth')
print(f'model2 {model2}')