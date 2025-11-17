
import os
import cv2

import torch
import torchvision.transforms as transforms


class ImageTransform():
    def __init__(self, resize=32, mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)  
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),  
                transforms.Normalize(mean, std) 
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


def trans_rand(imgs, labels, scale=0.25):
    bs = imgs.size()[0]
    cha = imgs.size()[1]
    height = imgs.size()[2]
    width = imgs.size()[3]
    trans = ImageTransform(height, (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

    lmax = labels.max()
    lmin = labels.min()
    bsz = lmax-lmin+1
    img_out = torch.zeros(bsz, cha, height, width)
    label_out = torch.zeros(bsz)

    k = 0
    imgs_max = imgs.max()
    imgs_min = imgs.min()
    for l in range(lmin, lmax+1):
        img = imgs_min+(imgs_max -imgs_min)*torch.rand(int(height*scale),int(width*scale),int(cha))
        img = cv2.resize( (img*255).numpy().astype('uint8') , (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        img = torch.tensor(img).permute(2,0,1).float()
        img = img/255
        img = trans.data_transform['test'](img)
        img_out[k] = img
        label_out[k] = l
        k = k + 1

    return img_out, label_out.long()    



def gen_keyimg_rand(dataset, outname, scale=0.25):

    x_tr, x_te, _, n_inputs, n_outputs, num_tasks_max, num_class, num_class_per_task, is_task_wise, is_perterb, img_size = utils.load_datasets(dataset)

    x_tr_lnd = x_tr
    x_te_lnd = x_te
    for tsk in range(len(x_tr)):
        x_tr_lnd[tsk][1], x_tr_lnd[tsk][2] = trans_rand(x_tr[tsk][1].view(-1,3,32,32), x_tr[tsk][2].view(-1), scale=0.25)
        x_te_lnd[tsk][1], x_te_lnd[tsk][2] = trans_rand(x_te[tsk][1].view(-1,3,32,32), x_te[tsk][2].view(-1), scale=0.25)

    scale = 0
    bit = 0
    torch.save([x_tr_lnd, scale, bit], outname)
    return 



if __name__ == '__main__':

    scale, bit = 0.25, 32
    gen_keyimg_rand(dataset='./cifar100_n2.pt', outname='./cifar_Rand4_n2.pt', scale=1/4)

