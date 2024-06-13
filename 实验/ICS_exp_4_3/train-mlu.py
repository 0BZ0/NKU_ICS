from torchvision.models import vgg19
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import torch_mlu
import cv2
import numpy
import os

os.putenv('MLU_VISIBLE_DEVICES','0')

class COCODataSet(Dataset):

    def __init__(self):
        super(COCODataSet, self).__init__()
        self.zip_files = ZipFile('./data/train2014.zip')
        self.data_set = []
        for file_name in self.zip_files.namelist():
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_files.read(file_path)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        
        image=cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        image=cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
       
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
        image = torch.from_numpy(image).float() / 255.0
        
        image = image.permute(2, 0, 1) 
        return image


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        a = vgg19(True)
        a = a.features
        self.layer1 = a[:4]
        self.layer2 = a[4:9]
        self.layer3 = a[9:18]
        self.layer4 = a[18:27]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4


class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            
            
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            
            nn.InstanceNorm2d(c),
          
            nn.ReLU(True),
            
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
        
            nn.InstanceNorm2d(c)
        )
        
    def forward(self, x):
     
        return x + self.layer(x)


class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            
          
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4, bias=False),
            
            nn.InstanceNorm2d(32),
            
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            
            nn.InstanceNorm2d(64),
            
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
           
            nn.InstanceNorm2d(128),
         
            nn.ReLU(True),

            
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),

            
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.InstanceNorm2d(64),
          
            nn.ReLU(True),

           
            nn.Upsample(scale_factor=2, mode='nearest'),
           
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
           
            nn.InstanceNorm2d(32),
            
            nn.ReLU(True),
            
           
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False),
           
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)


def load_image(path):
    
    image = cv2.imread(path)
   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    
    image = torch.from_numpy(image).float() / 255.0
   
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image


def get_gram_matrix(f_map):
  
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        f_map = f_map.reshape(n, c, h * w)
        gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
        return gram_matrix


if __name__ == '__main__':
    image_style = load_image('./data/udnie.jpg').cpu()
   
    mlu_image_style = image_style.to(torch.device('mlu'))
    net = VGG19().cpu()
    g_net = TransNet().cpu()
   
    mlu_g_net = g_net.to(torch.device('mlu'))
    
    mlu_net = net.to(torch.device('mlu'))
    print("mlu_net build PASS!\n")
   
    optimizer = torch.optim.Adam(mlu_g_net.parameters(), lr=0.001)
    
    loss_func = nn.MSELoss()
   
    mlu_loss_func = loss_func.to(torch.device('mlu'))
    print("build loss PASS!\n")
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")
    batch_size = 1
    data_loader = DataLoader(data_set, batch_size, True, drop_last=True)
    
    s1, s2, s3, s4 = mlu_net(mlu_image_style)
    
    
    s1 = get_gram_matrix(s1).detach()
    s2 = get_gram_matrix(s2).detach()
    s3 = get_gram_matrix(s3).detach()
    s4 = get_gram_matrix(s4).detach()    
    j = 1
    count = 0
    epochs = 1
    while j <= epochs:
        for i, image in enumerate(data_loader):
            image_c = image.cpu()
            
            mlu_image_c = image_c.to(torch.device('mlu'))
            
            mlu_image_g = mlu_g_net(mlu_image_c)
          
            out1, out2, out3, out4 = mlu_net(mlu_image_g)
          
            loss_s1 = mlu_loss_func(get_gram_matrix(out1), s1)
            loss_s2 = mlu_loss_func(get_gram_matrix(out2), s2)
            loss_s3 = mlu_loss_func(get_gram_matrix(out3), s3)
            loss_s4 = mlu_loss_func(get_gram_matrix(out4), s4)

          
            loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4

            
            c1, c2, c3, c4 = mlu_net(mlu_image_c)

            
            loss_c2 = mlu_loss_func(c2, out2.detach())
            loss_c = loss_c2

         
            loss = loss_c + 0.000000005 * loss_s

            
            optimizer.zero_grad()
           
            loss.backward()
            
            optimizer.step()
            print('j:',j, 'i:',i, 'loss:',loss.item(), 'loss_c:',loss_c.item(), 'loss_s:',loss_s.item())
            count += 1
            mlu_image_g = mlu_image_g.cpu()
            mlu_image_c = mlu_image_c.cpu()
            if i % 10 == 0:
                
                torch.save(mlu_g_net.state_dict(), './models/fst_train_mlu.pth')
               
                save_image(torch.cat((mlu_image_g, mlu_image_c), 3), './out/train_mlu/%d_%d.jpg' % (j, i))
        j += 1

print("MLU TRAIN RESULT PASS!\n")


