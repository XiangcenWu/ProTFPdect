import argparse, os, logging
from configs.config import MainConfig

from data_parse.custom_loader import *

from monai.transforms import *
from monai.losses import DiceLoss
# Compose, EnsureChannelFirstd, AdjustContrast, RandCropByPosNegLabeld, ScaleIntensityRangePercentilesd, SpatialPadd
from monai.networks.nets import UNETR
from training.train import train_net

from torch.optim import AdamW
from inference.inference import inference_net
def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    
    parser.add_argument("-c", "--config", help="The path of config file.", type=str)

    return parser.parse_args()


def config_log(result_dir, nick_name):
    FORMAT = '%(asctime)s, %(message)s'
    logging.basicConfig(
        filename=os.path.join(result_dir, nick_name+'.txt'),
        level=logging.INFO,
        filemode='w',
        format=FORMAT
    )
    logging.getLogger().addHandler(logging.StreamHandler())





def main():
    
    # cfg = MainConfig(args.config)
    

    
    transform = Compose([
        ReadH5Pkld(base_dir='data_converted'),
        # SpatialPadd(keys=['image', 'radio_positive', 'prostate', 'TP', 'FP'], spatial_size=(128, 128, 64)),
        RandCropByPosNegLabeld(keys=['image', 'radio_positive', 'prostate', 'TP', 'FP'], label_key="prostate", spatial_size=(128, 128, 32)),
        ScaleIntensityRangePercentilesd(['image'], 0, 100, 0, 1),
    ])
    inference_transform = Compose([
        Test_ReadH5Pkld(base_dir='data_converted'),
        # SpatialPadd(keys=['image', 'radio_positive', 'prostate', 'TP', 'FP'], spatial_size=(128, 128, 64)),
        ScaleIntensityRangePercentilesd(['image'], 0, 100, 0, 1),
    ])


    train, test = spilt_train_test()
    loader = get_loader(test[:1], transform=transform, batch_size=1, shuffle=True, drop_last=True)
    inference_loader = get_loader(test[:10], transform=inference_transform, batch_size=1, shuffle=True, drop_last=True)

    
    
    
    model = UNETR(in_channels=2, out_channels=2, img_size=(128, 128, 32), )
    loss_function = DiceLoss(sigmoid=True)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    for i in range(100):
        # loss = train_net(
        #     model=model,
        #     train_loader=loader,
        #     train_optimizer=optimizer,
        #     train_loss=loss_function,
        #     device='cuda:0'
        # )
        # print(loss)
        print(inference_net(model, inference_loader, 'cuda:0'))

    


    
    
if __name__ == "__main__":
    # args = parse_args()
    main()