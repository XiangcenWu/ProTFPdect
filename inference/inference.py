import torch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete





def inference_net(
        model, 
        inference_loader,
        device='cpu',
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.eval()
    
    num_TP, num_FP = 0, 0
    dect_TP, dect_FP = 0, 0

    for batch in inference_loader:

        img, radio_positive = batch["image"].to(device), batch["radio_positive"].to(device)
        # TP_gt, FP_gt = batch["TP"].to(device), batch["FP"].to(device)
        TP_FP = batch['TP_FP']
        TP_index, FP_index = get_TP_FP_index_gt(TP_FP)
        
        
        inference_img = concate_mri_radio_positive(img, radio_positive)

        
        # forward pass and calculate the selection


        # forward pass of selected data
        with torch.no_grad():
            inference_outputs = model(inference_img)

        inference_outputs = post_process(inference_outputs)
        
        
        if TP_index.shape[0] != 0:
            num_TP += 1
            
        
            TP_prediction = inference_outputs[TP_index[:, 0], TP_index[:, 1], TP_index[:, 2]]
            num_TP_pred = torch.sum(TP_prediction == 1).item()
            num_TP_gt = torch.sum(TP_FP == 1).item()
            
            if num_TP_pred / num_TP_gt > 0.5:
                dect_TP += 1
                


        if FP_index.shape[0] != 0:   
            num_FP += 1
            
             
            FP_prediction = inference_outputs[FP_index[:, 0], FP_index[:, 1], FP_index[:, 2]]
    
            num_FP_pred = torch.sum(FP_prediction == 2).item()
            num_FP_gt = torch.sum(TP_FP == 2).item()
            
            
            if num_FP_pred / num_FP_gt > 0.5:
                dect_FP += 1
                
                
        return dect_TP/num_TP, dect_FP/num_FP

        
        
        
def get_TP_FP_index_gt(gt):
    TP_index = torch.argwhere(gt == 1)
    FP_index = torch.argwhere(gt == 2)
    
    return TP_index, FP_index


def check_detected(prediction, location) -> bool:
    

    prediction = prediction.to('cpu')
    prediction = torch.argmax(prediction, dim=1).squeeze(0)
    prediction = prediction[location[0, :, 0], location[0, :, 1], location[0, :, 2]]
    # print(location.shape[1])
    # print(f'unique: {torch.unique(prediction)}')

    
    num_TP_pred = torch.sum(prediction == 1).item()
    
    num_FP_pred = torch.sum(prediction == 2).item()
    print(num_TP_pred, num_FP_pred)
    

    if num_TP_pred >= num_FP_pred:
        return 1
    else:
        return 0
    
    
    
    
    
    


def concate_mri_radio_positive(mri, radio_positive):
    return torch.cat([mri, radio_positive], dim=1)



def post_process(output):
    return torch.argmax(output, dim=1)