from utils import try_gpu
g_device = try_gpu()
def global_setting(trainning_set = 0.9,test_set = 0.1,num_pca=7,
                  num_emp = 3,train_round = 8,test_batch_size = 1024,train_batch_size = 320,
                  n_train_samples = 4, n_test_samples = None, c_map = False):
    return {
        'train_size':trainning_set,
        'test_size':test_set,
        'num_pca':num_pca,
        'num_emp':num_emp,
        'train_round':train_round,
        'n_train_samples':n_train_samples,
        'n_test_samples':n_test_samples,
        'src_dataset_name':'CK',
        'test_batch_size':test_batch_size,
        'c_map': c_map
    }

compare = {
    '3-D CNN+Extended-RAP':
        {
            'dropout':0.5,
            'started_lr':0.0003,
            'amp':True,
            'model':'3-D CNN',
            'patch_size':5,
            'num_epoch' :1000,
            'batch_size':1024,
            'test_during_training':0,
            'test_round': 10,
            'enable_rap':True,
            'augs':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1,'yflip':1,'transpose':1,'shearx':1,'sheary':1,'srotate':1},
            'aug_args':{'shear_max':0.2,'xint_max':0.1,'scale_std':0.1,'xfrac_std':0.1,'cutout_size':0.5,'aniso_std':0.3},
            'num_emb': 960,
            'knn':10,
            'select': 'max'
        },
        'HSI-CNN+Extended-RAP':
        {
            'dropout':0.5,
            'started_lr':0.0003,
            'amp':True,
            'model':'HSI-CNN',
            'patch_size':3,
            'num_epoch' :1000,
            'batch_size':1024,
            'test_during_training':0,
            'test_round': 10,
            'enable_rap':True,
            'augs':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1,'yflip':1,'transpose':1,'shearx':1,'sheary':1,'srotate':1},
            'aug_args':{'shear_max':0.2,'xint_max':0.1,'scale_std':0.1,'xfrac_std':0.1,'cutout_size':0.5,'aniso_std':0.3},
            'num_emb': 960,
            'knn':10,
            'select': 'max'
        },
        'HSI-CR+Extended-RAP':
        {
            'dropout':0.5,
            'started_lr':0.0003,
            'amp':True,
            'model':'HSI-CR',
            'patch_size':25,
            'num_epoch' :1000,
            'batch_size':1024,
            'test_during_training':0,
            'test_round': 10,
            'enable_rap':True,
            'augs':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1,'yflip':1,'transpose':1,'shearx':1,'sheary':1,'srotate':1},
            'aug_args':{'shear_max':0.2,'xint_max':0.1,'scale_std':0.1,'xfrac_std':0.1,'cutout_size':0.5,'aniso_std':0.3},
            'num_emb': 960,
            'knn':10,
            'select': 'max'
        },
        'EmbGCN':
        {
            'dropout':0.5,
            'started_lr':0.0003,
            'amp':True,
            'model':'EmbGCN',
            'patch_size':25,
            'num_epoch' :1000,
            'batch_size':1024,
            'test_during_training':0,
            'test_round': 10,
            'enable_rap':False,
            'augs':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1,'yflip':1,'transpose':1,'shearx':1,'sheary':1,'srotate':1},
            'aug_args':{'shear_max':0.2,'xint_max':0.1,'scale_std':0.1,'xfrac_std':0.1,'cutout_size':0.5,'aniso_std':0.3},
            'num_emb': 960,
            'knn':10,
            'select': 'max'
        },
        'EmbGCN+RAP':
        {
            'dropout':0.5,
            'started_lr':0.0003,
            'amp':True,
            'model':'EmbGCN',
            'patch_size':25,
            'num_epoch' :1000,
            'batch_size':1024,
            'test_during_training':0,
            'test_round': 10,
            'enable_rap':True,
            'augs':{'xflip':1,'xint':1,'scale':1,'rotate':1,'aniso':1,'xfrac':1,'cutout':1},
            'aug_args':{'shear_max':0.2,'xint_max':0.1,'scale_std':0.1,'xfrac_std':0.1,'cutout_size':0.5,'aniso_std':0.3},
            'num_emb': 960,
            'knn':10,
            'select': 'max'
        },
        'EmbGCN+Extended-RAP':
        {
            'dropout':0.5,
            'started_lr':0.0003,
            'amp':True,
            'model':'EmbGCN',
            'patch_size':25,
            'num_epoch' :1000,
            'batch_size':1024,
            'test_during_training':0,
            'test_round': 10,
            'enable_rap':True,
            'augs':{'xflip':1,'yflip':1,'transpose':1,'sflip':1,'rotate90':1,'shearx':1,'sheary':1,'scale':1,'aniso':1,'srotate':1,'rotate':1,'cutout':1},
            'aug_args':{'shear_max':0.2,'xint_max':0.2,'scale_std':0.3,'xfrac_std':0.1,'cutout_size':0.5,'aniso_std':0.5},
            'num_emb': 960,
            'knn':10,
            'select': 'max'
        }
        
}