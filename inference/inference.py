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
    num_detected_TP, num_detected_FP = 0, 0
    for batch in inference_loader:

        img, radio_positive = batch["image"].to(device), batch["radio_positive"].to(device)
        TP_gt, FP_gt = batch["TP"].to(device), batch["FP"].to(device)
        loaction = batch['position']
        
        inference_img = concate_mri_radio_positive(img, radio_positive)

        
        # forward pass and calculate the selection


        # forward pass of selected data
        with torch.no_grad():
            inference_outputs = sliding_window_inference(inference_img, (128, 128, 32), 1, model)
            print(len(torch.unique(inference_outputs)))
        inference_outputs = post_process(inference_outputs)

        
        
        TP_pred, FP_pred = inference_outputs[0, 0, :, :, :], inference_outputs[0, 1, :, :, :]
        TP_location_list, FP_location_list = loaction['TP'], loaction['FP']
        
        
        for TP_location in TP_location_list:
            num_TP += 1
            if check_detected(TP_pred, FP_pred, TP_gt, TP_gt, TP_location):
                num_detected_TP += 1
            
            
        for FP_location in FP_location_list:
            num_FP += 1
            if check_detected(FP_pred, FP_gt, FP_location):
                num_detected_FP += 1
                
                
    return  num_TP, num_FP, num_detected_TP, num_detected_FP



def check_detected(prediction, gt, location) -> bool:

    prediction, gt = prediction.to('cpu'), gt.to('cpu')
    
    prediction = prediction[location[0, :, 0], location[0, :, 1], location[0, :, 2]]
    
    num_ones_pred = torch.sum(prediction == 1).item()
    
    num_ones_gt = torch.sum(gt == 1).item()
    
    if num_ones_pred/num_ones_gt > 0.5:
        return True
    else:
        return False
    
    
    
    
    
    


def concate_mri_radio_positive(mri, radio_positive):
    return torch.cat([mri, radio_positive], dim=1)



def post_process(output):
    th = AsDiscrete(threshold=0.5)
    output = torch.sigmoid(output)
    return th(output)