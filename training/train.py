
import torch




def train_net(
        model, 
        train_loader,
        train_optimizer,
        train_loss,
        device='cpu',
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.train()
    
    _step = 0.
    _loss = 0.
    for batch in train_loader:

        img, radio_positive = batch["image"].to(device), batch["radio_positive"].to(device)
        TP_FP = batch["TP_FP"].to(device)
        
        train_img = concate_mri_radio_positive(img, radio_positive)
        # forward pass and calculate the selection


        # forward pass of selected data
        output = model(train_img)
        
        loss = train_loss(output, TP_FP)

        

        loss.backward()
        train_optimizer.step()
        train_optimizer.zero_grad()

        _loss += loss.item()
        _step += 1.
    _epoch_loss = _loss / _step

    return _epoch_loss



def concate_mri_radio_positive(mri, radio_positive):
    return torch.cat([mri, radio_positive], dim=1)


# def concate_TP_FP(TP, FP):
#     return torch.cat([TP, FP], dim=1)