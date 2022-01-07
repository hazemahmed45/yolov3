import torch
from torch import nn
from config import ARCHI

class CNNBlock(nn.Module):
    def __init__(self,in_c,out_c,bn_act=True,**kwargs) :
        super().__init__()
        self.conv=nn.Conv2d(in_channels=in_c,out_channels=out_c,bias=not bn_act,**kwargs)
        self.bn=nn.BatchNorm2d(out_c)
        self.leaky=nn.LeakyReLU(0.1)
        self.use_bn_act=bn_act
    def forward(self,x):
        if(self.use_bn_act):
            return self.leaky(self.bn(self.conv(x)))
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self,channels,use_residual=True,num_repeat=1) -> None:
        super().__init__()
        self.layers=nn.ModuleList()
        for _ in range(num_repeat):
            self.layers.append(nn.Sequential(
                CNNBlock(channels,channels//2,kernel_size=1),
                CNNBlock(channels//2,channels,kernel_size=3,padding=1)
            ))
        self.use_residual=use_residual
        self.num_repeat=num_repeat
    def forward(self,x):
        # print(x.shape)
        for layer in self.layers:
            if(self.use_residual):
                x=layer(x)+x
            else:
                x=layer(x)
        return x
    
class ScalePrediction(nn.Module):
    def __init__(self,in_c,num_classes):
        super().__init__()
        self.pred=nn.Sequential(
            CNNBlock(in_c,2*in_c,kernel_size=3,padding=1),
            CNNBlock(2*in_c,(num_classes+5)*3,bn_act=False,kernel_size=1)
        )
        self.num_classes=num_classes
    def forward(self,x):
        return self.pred(x).reshape(x.shape[0],3,self.num_classes+5,x.shape[2],x.shape[3]).permute(0,1,3,4,2)
    
class YOLOv3(nn.Module):
    def __init__(self,config,in_c=3,num_classes=20) -> None:
        super().__init__()
        self.num_classes=num_classes
        self.in_c=in_c
        self.config=config
        self.layers=self._create_conv_layers()
    def forward(self,x):
        outputs=[]
        route_connections=[]
        for layer in self.layers:

            if(isinstance(layer,ScalePrediction)):
                outputs.append(layer(x))
                continue
            x=layer(x)
            if(isinstance(layer,ResidualBlock) and layer.num_repeat==8):
                route_connections.append(x)
            elif (isinstance(layer,nn.Upsample)):
                x=torch.cat([x,route_connections[-1]],dim=1)
                route_connections.pop()
        return outputs
    def _create_conv_layers(self):
        layers=nn.ModuleList()
        in_c=self.in_c
        for module_cfg in self.config:
            if(isinstance(module_cfg,tuple)):
                out_c,k,s=module_cfg
                layers.append(CNNBlock(
                    in_c,
                    out_c,
                    kernel_size=k,
                    stride=s,
                    padding=1 if k==3 else 0
                ))
                in_c=out_c
            elif(isinstance(module_cfg,list)):
                num_repeat=module_cfg[1]
                layers.append(ResidualBlock(in_c,num_repeat=num_repeat))
            elif(isinstance(module_cfg,str)):
                if(module_cfg=='S'):
                    layers+=[
                        ResidualBlock(in_c,use_residual=False,num_repeat=1),
                        CNNBlock(in_c=in_c,out_c=in_c//2,kernel_size=1),
                        ScalePrediction(in_c=in_c//2,num_classes=self.num_classes)
                    ]
                    in_c=in_c//2
                elif(module_cfg=='U'):
                    layers.append(nn.Upsample(scale_factor=2))
                    in_c=in_c*3
        return layers
    
if __name__ == '__main__':
    num_classes=20
    IMG_SIZE=416
    model=YOLOv3(config=ARCHI,num_classes=num_classes)
    x=torch.randn((2,3,IMG_SIZE,IMG_SIZE))
    out=model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    assert model(x)[0].shape == (2, 3, IMG_SIZE//32, IMG_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMG_SIZE//16, IMG_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMG_SIZE//8, IMG_SIZE//8, num_classes + 5)
    print("Success!")