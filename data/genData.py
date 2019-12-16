import torchvision
import torchvision.transforms as transforms

class genCifar10():
    def __init__(self, dir_root, is_train=False, transform=False, download=False): 
        
        self.dir_root = dir_root
        self.is_train = is_train
        #self.download = download
        self.transform = transform
    
    def transform_data(self):
        if(self.is_train):
            return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), # change Tensor
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        else:
            return transforms.Compose([
                transforms.ToTensor(), # change Tensor
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
    
    def dataLoader(self):
        if(self.transform):
            dataset = torchvision.datasets.CIFAR10(root=self.dir_root, train=self.is_train, transform=self.transform_data(), download=True)
        else:
            dataset = torchvision.datasets.CIFAR10(root=self.dir_root, train=self.is_train, download=True)

        return dataset



    
if __name__=="__main__":
    #pass
    dataSet_train = genCifar10("./cifar10", is_train=True, transform=True)
    train_data = dataSet_train.dataLoader()
    
    for idx, batch in enumerate(train_data):
        
        data, label = batch
        print('data shape:', data.shape)    # B C W H
        print('label:', label[0])     #
    
    
    
    
    