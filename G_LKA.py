import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# 1. 하이퍼파라미터 설정
batch_size    = 128
learning_rate = 0.001
num_epochs    = 20

# 2. 데이터 전처리 및 DataLoader 준비
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_loader = DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
    batch_size=batch_size, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
    batch_size=batch_size, shuffle=False, num_workers=4
)

# 3. GhostModule 정의
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels  = init_channels * (ratio - 1)

        # primary: 일반 Conv
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride,
                      padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        # cheap: depthwise Conv
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels,
                      dw_size, 1, dw_size//2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

# 4. LKA 모듈 정의 (GhostModule 통합)
class LKA(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(LKA, self).__init__()
        self.in_ch  = in_channels
        self.out_ch = out_channels or in_channels

        # 1) Depthwise conv: 공간 어텐션 맵 추출 (in_ch → in_ch)
        self.dw1 = nn.Conv2d(self.in_ch, self.in_ch,
                             kernel_size=5, padding=2,
                             groups=self.in_ch, bias=False)
        self.dw2 = nn.Conv2d(self.in_ch, self.in_ch,
                             kernel_size=7, padding=9, dilation=3,
                             groups=self.in_ch, bias=False)

        # 2) GhostModule로 채널 확장 및 정제 (in_ch → out_ch)
        self.ghost = GhostModule(self.in_ch, self.out_ch,
                                 kernel_size=1, ratio=2, dw_size=3, relu=False)

        # 3) Skip branch: in_ch → out_ch
        if self.in_ch != self.out_ch:
            self.skip = nn.Conv2d(self.in_ch, self.out_ch,
                                  kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # a) spatial attention 맵 생성
        attn = self.dw1(x)
        attn = self.dw2(attn)
        # b) GhostModule로 채널 확장 & 정제
        attn = self.ghost(attn)           # [B, out_ch, H, W]

        # c) skip branch
        identity = self.skip(x)           # [B, out_ch, H, W]

        # d) 어텐션 + residual
        out = identity + attn             # [B, out_ch, H, W]
        return out

# 5. 네트워크 정의
class SimpleCNN_LKA_RF23(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_LKA_RF23, self).__init__()
        self.features = nn.Sequential(
            LKA(3,   64),  nn.BatchNorm2d(64),  nn.ReLU(inplace=True),  nn.MaxPool2d(2),
            LKA(64, 128),  nn.BatchNorm2d(128), nn.ReLU(inplace=True),  nn.MaxPool2d(2),
            LKA(128,256),  nn.BatchNorm2d(256), nn.ReLU(inplace=True),  nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    
            nn.Linear(256*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 6. 디바이스 설정 및 모델 초기화
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = SimpleCNN_LKA_RF23().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 7. 학습 함수
def train_one_epoch(epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total   += targets.size(0)
        correct += pred.eq(targets).sum().item()

    print(f"[Train] Epoch {epoch+1}/{num_epochs}  "
          f"Loss: {running_loss/total:.4f}  Acc: {100.*correct/total:.2f}%")

# 8. 평가 함수
def evaluate():
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total   += targets.size(0)
            correct += pred.eq(targets).sum().item()

    print(f"[Test ] Loss: {running_loss/total:.4f}  Acc: {100.*correct/total:.2f}%")
    return 100.*correct/total

# 9. 학습 & 평가 루프
best_acc = 0.0
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    acc = evaluate()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f">>> New best model saved (Acc: {best_acc:.2f}%)\n")
    else:
        print()

print(f"최종 최고 정확도: {best_acc:.2f}%")

# 10. 프로파일링 (파라미터, FLOPs, FPS)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")
try:
    from thop import profile
    flops, _ = profile(model, inputs=(torch.randn(1,3,32,32).to(device),), verbose=False)
    print(f"FLOPs: {flops:,}")
except ImportError:
    print("thop 패키지 설치: pip install thop")

model.eval()
with torch.no_grad():
    N = 100
    dummy = torch.randn(batch_size, 3, 32, 32).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"FPS: {N*batch_size/elapsed:.2f} samples/sec")
