from tqdm import tqdm
import torch.nn.functional as F
from trainer.train_utils import *
import math,random


def trainer_fixmatch(model, teacher, data_loader, optimizer, criterion, c_w, alpha, epoch_num):
    global_step = epoch_num * len(data_loader)
    model.train()
    teacher.train()
    class_loss_tracker = AverageMeter()
    consi_loss_tracker = AverageMeter()
    t_class_loss_tracker = AverageMeter()
    
    for i, (X, Y_true, Y_mask) in tqdm(enumerate(data_loader)):

        #print(' --- start mt ---')
        X = X.cuda()
        Y_true = Y_true.cuda()
        Y_mask = Y_mask.cuda()

        #weakly aug
        X_aug = tfmask(X, 2, 10)
        outputs = model(X_aug)
        # print('outputted')
        
        # Regardless of what criterion or whether this is instrument-wise
        # Let the criterion function deal with it
        class_loss = criterion(outputs[Y_mask], Y_true[Y_mask])
        
        # Compute consistency loss here
        #strongly aug
        #tfmask
        X_aug = tfmask(X, 3, 50)
        outputs_ = teacher(X_aug)
        outputs = binarize_targets(outputs, 0.7)
        
        consistency_loss = criterion(outputs, outputs_)
        
        #train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss_xe + wu * loss_xeu + wd * loss_wd, colocate_gradients_with_ops=True)
        loss = class_loss + c_w*consistency_loss

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        # Update teacher
        global_step += 1
        update_ema_variables(model, teacher, global_step, alpha)
        t_class_loss = criterion(outputs_[Y_mask], Y_true[Y_mask])
                                 
        # Update average meters
        class_loss_tracker.update(class_loss.item())
        consi_loss_tracker.update(consistency_loss.item())
        t_class_loss_tracker.update(t_class_loss.item())
    return (class_loss_tracker.avg, consi_loss_tracker.avg, t_class_loss_tracker.avg)



def tfmask(X, t, f):
    
    #(20000, 10, 128)
    tmask_length = random.randint(1, t)
    tmask_start = random.randint(0, 7)
    X[:,tmask_start:tmask_start+tmask_length,:] = 0.
    
    fmask_length = random.randint(2, f)
    fmask_start = random.randint(0, 70)
    X[:,:,fmask_start:fmask_start+fmask_length] = 0.
    
    return X

def binarize_targets(targets, threshold=0.5):
    targets[targets < threshold] = 0
    targets[targets > 0] = 1
    return targets