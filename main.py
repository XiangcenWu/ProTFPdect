import argparse, os, logging
from configs.config import MainConfig

from data_parse.custom_loader import *

from monai.transforms import *
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss
# Compose, EnsureChannelFirstd, AdjustContrast, RandCropByPosNegLabeld, ScaleIntensityRangePercentilesd, SpatialPadd
from monai.networks.nets import UNETR
from monai.networks.nets.swin_unetr import SwinUNETR
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
        # RandCropByPosNegLabeld(keys=['image', 'radio_positive', 'prostate', 'TP_FP'], label_key="prostate", spatial_size=(128, 128, 32), pos=1, neg=1, num_samples=2),
        ScaleIntensityRangePercentilesd(['image'], 0, 100, 0, 1),
    ])
    inference_transform = Compose([
        Test_ReadH5Pkld(base_dir='data_converted'),
        # SpatialPadd(keys=['image', 'radio_positive', 'prostate', 'TP', 'FP'], spatial_size=(128, 128, 64)),
        ScaleIntensityRangePercentilesd(['image'], 0, 100, 0, 1),
    ])


    train, test = spilt_train_test(seed=2)
    loader = get_loader(test[:1], transform=transform, batch_size=1, shuffle=True, drop_last=True)
    inference_loader = get_loader(test[:1], transform=inference_transform, batch_size=1, shuffle=True, drop_last=True)

    
    
    
    model = SwinUNETR(in_channels=2, out_channels=3, img_size=(128, 128, 64))
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    for i in range(40):
        loss = train_net(
            model=model,
            train_loader=loader,
            train_optimizer=optimizer,
            train_loss=loss_function,
            device='cuda:0'
        )
        print(loss)

        # print(inference_net(model, inference_loader, 'cuda:0'))
        
        
        ################

    import matplotlib.pyplot as plt
    
    for i in range(5):
        data = next(iter(inference_loader))
        train_img = torch.cat([data['image'], data['radio_positive']], dim=1).to('cuda:0')
        with torch.no_grad():
            inference_outputs = sliding_window_inference(train_img, (128, 128, 32), 1, model)
        inference_outputs = post_process(inference_outputs)
        plt.imshow(inference_outputs[0, :, :, 25+2*i].cpu().detach())
        plt.show()


    

def post_process(output):
    return torch.argmax(output, dim=1)
    
    
if __name__ == "__main__":
    # args = parse_args()
    main()