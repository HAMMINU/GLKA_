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

# 2. 데이터 전처리 및 데이터셋 준비
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

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_dataset  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                          shuffle=False, num_workers=4)

# 3. LKA 모듈 정의
class LKA(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(LKA, self).__init__()
        self.in_ch  = in_channels
        self.out_ch = out_channels or in_channels

        # 1) 23×23 Conv → 채널 확장(attn)
        self.conv23 = nn.Conv2d(
            self.in_ch, self.out_ch,
            kernel_size=23, padding=11, bias=False
        )
        # 2) Pointwise conv (채널 유지)
        self.pw = nn.Conv2d(
            self.out_ch, self.out_ch,
            kernel_size=1, bias=False
        )

        # 3) Skip branch: in_ch→out_ch (필요 시)
        if self.in_ch != self.out_ch:
            self.skip = nn.Conv2d(
                self.in_ch, self.out_ch,
                kernel_size=1, bias=False
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # a) 23×23 conv + PW
        attn = self.conv23(x)        # [B, out_ch, H, W]
        attn = self.pw(attn)         # [B, out_ch, H, W]

        # b) skip branch
        identity = self.skip(x)      # [B, out_ch, H, W]

        # c) residual sum
        out = identity + attn        # [B, out_ch, H, W]
        return out

# 4. 모델 정의
class SimpleCNN_LKA_RF23(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_LKA_RF23, self).__init__()
        self.features = nn.Sequential(
            # Block1: receptive field ≈23×23 & 3→64 채널 확장
            LKA(in_channels=3,  out_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32→16

            # Block2: RF 확장 & 64→128
            LKA(in_channels=64, out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16→8

            # Block3: RF 확장 & 128→256
            LKA(in_channels=128, out_channels=256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),           # (batch, 256*4*4)
            nn.Linear(256*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 5. 디바이스 설정 및 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SimpleCNN_LKA_RF23().to(device)

# 6. 손실함수 및 최적화기 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 7. 학습 함수
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = 100. * correct / total
    print(f"[Train] Epoch {epoch+1}/{num_epochs}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%")

# 8. 평가 함수
def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = 100. * correct / total
    print(f"[Test ] Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%")
    return epoch_acc

# 9. 전체 학습 및 평가 루프
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

# 10. 모델 프로파일링
# 10-1) 파라미터 수
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# 10-2) FLOPs 계산 (thop 필요)
try:
    from thop import profile
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"FLOPs: {flops:,}")
except ImportError:
    print("thop 패키지 설치: pip install thop")

# 10-3) FPS 측정
model.eval()
with torch.no_grad():
    N = 100
    dummy_batch = torch.randn(batch_size, 3, 32, 32).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(N):
        _ = model(dummy_batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    fps = N * batch_size / elapsed
    print(f"FPS: {fps:.2f} samples/sec")
