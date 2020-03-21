class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.09) 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=35, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.09) 
        )
        self.pool = nn.MaxPool2d(2, 2) 
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.09) 
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.09) 
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.09) 
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.09) 
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.09) 
        )
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.09) 
        )

        self.avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        ) 
        self.conv6_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )
        self.convblock_out = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # ou
        self.linear = nn.Linear(in_features=512,out_features=10, bias=False)

    def forward(self, x):
        x1 = x
        x2 = self.conv2(x1) #32 output
        x3 = self.conv3(torch.cat((x1,x2),dim=1)) #35 input channels 64 output channels
        x4 = self.pool(x3) #64 output channels
        x5 = self.convblock5(x4) #128 output channels 
        x5 = self.conv1_1(x5) #64 channels
        x6 = self.convblock6(torch.cat((x4 , x5),dim=1)) #64,64 input channels 256 output channels
        x6 = self.conv2_1(x6) #64 channels
        x7 = self.convblock7(torch.cat((x4, x5 , x6), dim=1)) #256 output channels
        x7 = self.conv3_1(x7) #64 output
        x8 = self.pool(torch.cat((x5 , x6 , x7),dim=1)) #64,64,64 input channels 192 output channels
        x9 = self.convblock8(x8) #192 input channels , 256 output
        x9 = self.conv4_1(x9) #64 output
        x10 = self.convblock9(torch.cat((x8 , x9),dim=1)) #256 input channels 256 output channels
        x11 = self.convblock11(torch.cat((x8 , x9 , x10),dim=1)) #192,64,256 == 512 inputchannels, output 256
        x12 = self.avg(x11) #1,1,512
        x12 = self.conv6_1(x12)
        x12 = x12.view(-1, 10)
        return F.log_softmax(x12)