import os
import numpy as np
import torch
import torch.utils.data as utils
from torchvision import transforms
import pickle

from .utility_functions import audio_image_csv_to_dict, load_image

class CustomAudioVisualDataset(utils.Dataset):
    def __init__(self, audio_predictors, audio_target, image_path=None, image_audio_csv_path=None, transform_image=None):
        self.audio_predictors = audio_predictors[0]
        self.audio_target = audio_target
        self.audio_predictors_path = audio_predictors[1]
        self.image_path = image_path
        if image_path:
            print("AUDIOVISUAL ON")
            self.image_audio_dict = audio_image_csv_to_dict(image_audio_csv_path)
            self.transform = transform_image
        else:
            print("AUDIOVISUAL OFF")
    
    def __len__(self):
        return len(self.audio_predictors)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_pred = self.audio_predictors[idx]
        audio_trg = self.audio_target[idx]
        audio_pred_path = self.audio_predictors_path[idx]
        
        if self.image_path:
            image_name = self.image_audio_dict[audio_pred_path]
            img = load_image(os.path.join(self.image_path, image_name))
            
            if self.transform:
                img = self.transform(img)

            return (audio_pred, img), audio_trg
        
        return audio_pred, audio_trg

    
# class CustomBatch:
#     def __init__(self, data):
#         transposed_data = list(zip(*data))
#         transposed_data_0 = list(zip(*transposed_data[0]))
#         self.audio_pred = torch.stack(transposed_data_0[0], 0)
#         self.inp = list(zip(self.audio_pred, transposed_data_0[1]))
#         self.tgt = torch.stack(transposed_data[1], 0)

#     # custom memory pinning method on custom type
#     def pin_memory(self):
#         self.audio_pred = self.audio_pred.pin_memory()
#         self.tgt = self.tgt.pin_memory()
#         return self

# def collate_wrapper(batch):
#     return CustomBatch(batch)


def load_dataset(args):
    #LOAD DATASET
    print ('\nLoading dataset')

    with open(args.training_predictors_path, 'rb') as f:
        training_audio_predictors = pickle.load(f)
    with open(args.training_target_path, 'rb') as f:
        training_target = pickle.load(f)
    '''with open(args.validation_predictors_path, 'rb') as f:
        validation_audio_predictors = pickle.load(f)
    with open(args.validation_target_path, 'rb') as f:
        validation_target = pickle.load(f)
    with open(args.test_predictors_path, 'rb') as f:
        test_predictors = pickle.load(f)
    with open(args.test_target_path, 'rb') as f:
        test_target = pickle.load(f)'''
    
    training_img_predictors = training_audio_predictors[1]
    training_audio_predictors = np.array(training_audio_predictors[0])
    training_target = np.array(training_target)
    #validation_img_predictors = validation_audio_predictors[1]
    #validation_audio_predictors = np.array(validation_audio_predictors[0])
    # validation_img_predictors = validation_predictors[1]
    #validation_target = np.array(validation_target)
    #test_audio_predictors = np.array(test_predictors[0])
    #test_img_predictors = test_predictors[1]
    #test_target = np.array(test_target)

    print ('\nShapes:')
    print ('Training predictors: ', training_audio_predictors.shape)
    #print ('Validation predictors: ', validation_audio_predictors.shape)
    #print ('Test predictors: ', test_audio_predictors.shape)

    #convert to tensor
    training_audio_predictors = torch.tensor(training_audio_predictors).float()
    #validation_audio_predictors = torch.tensor(validation_audio_predictors).float()
    #test_audio_predictors = torch.tensor(test_audio_predictors).float()
    training_target = torch.tensor(training_target).float()
    #validation_target = torch.tensor(validation_target).float()
    #test_target = torch.tensor(test_target).float()
    
    #build dataset from tensors
    # tr_dataset = utils.TensorDataset(training_predictors, training_target)
    # val_dataset = utils.TensorDataset(validation_predictors, validation_target)
    # test_dataset = utils.TensorDataset(test_predictors, test_target)
    
    transform = transforms.Compose([  
        transforms.ToTensor(),
    ])

    tr_dataset = CustomAudioVisualDataset((training_audio_predictors, training_img_predictors), training_target, args.path_images, args.path_csv_images_train, transform)
    #val_dataset = CustomAudioVisualDataset((validation_audio_predictors,validation_img_predictors), validation_target, args.path_images, args.path_csv_images_train, transform)
    #test_dataset = CustomAudioVisualDataset((test_audio_predictors,test_img_predictors), test_target, args.path_images, args.path_csv_images_test, transform)

    return tr_dataset