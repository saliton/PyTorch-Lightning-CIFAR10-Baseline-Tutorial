[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Soliton-Analytics-Team/PyTorch-Lightning-CIFAR10-Baseline-Tutorial/blob/main/PyTorch_Lightning_CIFAR10チュートリアル解説.ipynb)

# PyTorch Lightning CIFAR10 Baseline Tutorial を解説

PyTorch LightningはPyTorchの色々と細かい点を隠蔽して、オレオレ実装になりがちな学習ルーチンを標準化してくれます。そのため、コードが比較的読みやすくなるという利点があります。今回、[ここ](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/notebooks/07-cifar10-baseline.ipynb)にPyTorch LightningをCIFAR10に適用するnotebookを見つけましたので、これを元に解説します。実際にGoogle Colabで実行できるようにしてありますので、是非試してください。

## Setup

まずは、結果を保存するためにgoogle driveをマウントし、ディレクトリを作ってそこに移動します。


```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/CIFAR10
%cd /content/drive/MyDrive/CIFAR10
```

次にPyTorch Lightningをインストールします。pipで簡単です。[lightning-bolts](https://github.com/PyTorchLightning/lightning-bolts/)は、いろいろ便利な小道具なりモデルなりを集めたモジュールです。今回はCIFAR10のDataModuleを使うためにインストールしています。


```python
! pip install pytorch-lightning lightning-bolts -qU
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
```

pytorch_lightningには諸々の乱数シードを一発で設定するメソッドがあるので、それを使用します。これで、学習用と評価用のデータの切り分けなどは毎回同じにできます。ただ、GPUの実行タイミングは制御できないので、学習結果についてはばらつきがあります。


```python
pl.seed_everything(7)
```

## CIFAR10 Data Module

バッチサイズを設定します。元のチュートリアルのバッチサイズは32でしたが、256でも大丈夫そうなので、256にしてみました。もししょぼいGPUを割り当てられたりして、メモリアロケーションエラーになるようでしたら、この数字を小さくしてみてください。


```python
batch_size = 256
```

入力画像に対する画像変換を定義します。学習用にはオーグメンテーションを設定していますが、評価用にはテンソル変換と標準化のみ設定しています。


```python
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])
```

boltsに用意されているCIFAR10DataModuleを利用してデータモジュールを作成します。


```python
cifar10_dm = CIFAR10DataModule(
    batch_size=batch_size,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
    num_workers=2
)
```

実際、独自データを学習させる現場では、このデータモジュールを定義するのが最もオリジナルな部分であることが多いのですが、ここではboltsを利用して端折っています。データモジュールの作成に関しては別記事に譲りたいと思います。

## Resnet

次に、torchvisionに用意されているresnet18を利用してモデルを作成する関数を定義します。

```python
def create_model():
    model = torchvision.models.resnet18(pretrained=True)
    model.fc.out_features = 10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model
```

実は、元のチュートリアルは
```python
model = torchvision.models.resnet18(pretrained=False, num_classes=10)
```
となっていて学習済み重みを使っていません。これを学習済み重みを使うようにしたいと思い、
```python
model = torchvision.models.resnet18(pretrained=True, num_classes=10)
```
とすると、構造が違うというエラーがでてしまいます。torchvisionのResnetはImageNetによる事前学習なので、出力のクラス数は1000です。このクラス数を変更するのがnum_classes引数なのですが、これを変えると学習済みの重みをロードできない仕様のようです。

そこで、まずはnum_classesを指定せずにpretrained=Trueで学習済み重みをロードしてから出力クラス数を変更するようにしました。
```python
model = torchvision.models.resnet18(pretrained=True)
model.fc.out_features = 10
```
ここで登場するmodel.fc.out_featuresはどこから現れたのかと思われるかもしれませんが、これはmodelをprintすると表示される構造の最後のところから見つけました。
```python
ResNet(
    ...
    (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```

次にmodel.conv1とmodel.maxpoolです。これは元のチュートリアルにもあります。
```python
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()
```
これらはmodelの構造の最初のところにあるものです。元々は以下のようになっています。
```python
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  ...
```
これはImageNetの画像サイズ(224x224)に対応したものです。しかし、CIFAR10の画像サイズは(32x32)です。元のままではkernel_sizeもstrideも大雑把すぎます。そこでより細かくピクセルの情報を見るようにconv1を付け替えているのです。またmodel.maxpoolもこの時点で情報を集約しないようにIdentity()を設定して無効化しています。

このように、出来合いのモデルを使う場合、効果を発揮するためには入力の条件に合わせて細かい調整が必要です。ただ、モデルの構造を表示させてみれば意外とわかりやすいので、独自の調整を施して試してみてください。

## Lightning Module

LightningModuleはほぼテンプレートです。損失関数にF.nll_lossを使っていますが、お好みのものに取り替えても大丈夫でしょう。また最適化関数もSGDとOneCycleLRを使っています。この辺りも標準化とは離れて独自性が現れてしまいますね。ただ、損失値のbackward()を呼んだり、modelのeval()を呼んだりなどのpytorch由来の細かい注意をしなくて良いのは大変助かります。


```python
class LitResnet(pl.LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        steps_per_epoch = 45000 // batch_size
        scheduler_dict = {
            'scheduler': OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
```

```python
model = LitResnet(lr=0.05)

trainer = pl.Trainer(
    progress_bar_refresh_rate=1,
    max_epochs=50,
    gpus=1,
    logger=pl.loggers.TensorBoardLogger('lightning_logs/', name='resnet'),
    callbacks=[LearningRateMonitor(logging_interval='step')],
)
```

```python
trainer.fit(model, cifar10_dm)
trainer.save_checkpoint("resnet18.ckpt")
```

```python
trainer.test(model, datamodule=cifar10_dm)
```

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

    --------------------------------------------------------------------------------
    DATALOADER:0 TEST RESULTS
    {'test_acc': 0.9458000063896179, 'test_loss': 0.216917023062706}
    --------------------------------------------------------------------------------

    [{'test_acc': 0.9458000063896179, 'test_loss': 0.216917023062706}]

元々のチュートリアルにも40-50エポックで93-94%と書いてありますので、いい感じで再現されているようです。

## Bounus: Use [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407) to get a boost on performance

SWAは簡単に使えて精度もあがるということなので、使ってみます。以下のように単純にラップするだけで使えます。


```python
class SWAResnet(LitResnet):
    def __init__(self, trained_model, lr=0.01):
        super().__init__()

        self.save_hyperparameters('lr')
        self.model = trained_model
        self.swa_model = AveragedModel(self.model)

    def forward(self, x):
        out = self.swa_model(x)
        return F.log_softmax(out, dim=1)

    def training_epoch_end(self, training_step_outputs):
        self.swa_model.update_parameters(self.model)

    def validation_step(self, batch, batch_idx, stage=None):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log(f'val_loss', loss, prog_bar=True)
        self.log(f'val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        return optimizer

    def on_train_end(self):
        update_bn(self.datamodule.train_dataloader(), self.swa_model, device=self.device)
```


```python
swa_model = SWAResnet(model.model, lr=0.01)
swa_model.datamodule = cifar10_dm

swa_trainer = pl.Trainer(
    progress_bar_refresh_rate=1,
    max_epochs=20,
    gpus=1,
    logger=pl.loggers.TensorBoardLogger('lightning_logs/', name='swa_resnet'),
)
```


```python
swa_trainer.fit(swa_model, cifar10_dm)
swa_trainer.save_checkpoint("swa_resnet18.ckpt")
```


```python
swa_trainer.test(model, datamodule=cifar10_dm)
```

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

    --------------------------------------------------------------------------------
    DATALOADER:0 TEST RESULTS
    {'test_acc': 0.9506000280380249, 'test_loss': 0.20038478076457977}
    --------------------------------------------------------------------------------

    [{'test_acc': 0.9506000280380249, 'test_loss': 0.20038478076457977}]


確かにちょっとだけ精度が向上していますが、誤差の範囲ではという気もしますね。

以下を実行するとtensorboardが表示されますので、学習経過などを観察してみてください。


```python
%reload_ext tensorboard
%tensorboard --logdir lightning_logs/
```

Enjoy!
