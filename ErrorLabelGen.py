import numpy as np
import pdb
import yaml
import os
import errno

with open("./config/semantic-kitti.yaml",'r') as f:
    conf = yaml.safe_load(f)
    learning_inv = conf['learning_map_inv']
    all_class = {v:k for k,v in learning_inv.items()}

# pdb.set_trace()


FRNET_path = "D:\FileFromRemote\ErrorMap\FRNet\sequences\\08\labels\\"
im_file = []


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def create(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

im_file += absoluteFilePaths(FRNET_path)

pdb.set_trace()
img =0.0
for i in range(len(im_file)):
    # pdb.set_trace()


    spfile = im_file[i].replace("FRNet","SphereFormer")
    pvkdfile= im_file[i].replace("FRNet","PVKD")
    gtfile = "D:\MyProject\Ridar_SS\Semantic-kitti\dataset\sequences"+"\\08\\labels\\"+im_file[i][-12:]
    # pdb.set_trace()

    frnet = np.fromfile(im_file[i],dtype=np.int32)
    sphere = np.fromfile(spfile,dtype=np.int32)
    pvkd = np.fromfile(pvkdfile,dtype=np.int32)
    gt = np.fromfile(gtfile,dtype=np.int32)&0xff
    # pdb.set_trace()
    #moving to origin
    gt[np.where(gt==252)] = 10
    gt[np.where(gt==253)] = 31
    gt[np.where(gt==254)] = 30
    gt[np.where(gt==255)] = 32
    gt[np.where(gt==256)] = 20
    gt[np.where(gt==257)] = 20
    gt[np.where(gt==258)] = 18
    gt[np.where(gt==259)] = 20
    gt[np.where((gt==99)|(gt==52)|(gt==1))] = 0
    unlabeled = np.where(gt==0)[0] 
    

    frnet[unlabeled] = 0
    sphere[unlabeled] = 0
    pvkd[unlabeled] = 0

    fr_error_index = np.where(frnet!=gt)[0]
    sp_error_index = np.where(sphere!=gt)[0]

    pvkd_error_index = np.where(pvkd!=gt)[0]

    def getErrorCount(err_index,error_array):
        ErrorCount = np.zeros((20,20),dtype=np.int32)
        error_sum = np.concatenate((err_index.reshape((-1,1)),error_array[err_index].reshape((-1,1)),gt[err_index].reshape((-1,1))),axis=1)
        
        unique_pred = np.unique(error_sum[:,1])
        # pdb.set_trace()
        for j,e in enumerate(unique_pred):
            # pdb.set_trace()
            for i,(k,v)in enumerate(all_class.items()):
                ErrorCount[all_class[e]][v] = len(np.where((error_sum[:,1]==e)&(error_sum[:,2]==k))[0])
            # pdb.set_trace()
        # pdb.set_trace()
        return ErrorCount

    fr_error_sum = getErrorCount(fr_error_index.copy(),frnet.copy())
    sp_error_sum = getErrorCount(sp_error_index,sphere)
    pvkd_error_sum = getErrorCount(pvkd_error_index,pvkd)
    # pdb.set_trace()

    frnet[fr_error_index] = 100
    frnet[np.where(frnet!=100)] = 101

    sphere[sp_error_index] = 100
    sphere[np.where(sphere!=100)] = 101

    pvkd[pvkd_error_index] = 100
    pvkd[np.where(pvkd!=100)] = 101
    # pdb.set_trace()
    gt_class = np.unique(gt)
    gt_classCount = dict()
    fr_classCount = dict()
    sphere_classCount = dict()
    pvkd_classCount = dict()

    for j in gt_class:
        gt_classCount[f'{i}'] = len(np.where(gt==j)[0])
        fr_classCount[f'{i}'] = len(np.where(frnet==j)[0])
        pvkd_classCount[f'{i}'] = len(np.where(pvkd==j)[0])
        sphere_classCount[f'{i}'] = len(np.where(sphere==j)[0])
    
    outfile = im_file[i].replace("labels","predictions")
    print(f'{outfile}')
    sp_outfile = outfile.replace("FRNet","SphereFormer")
    pvkd_outfile = outfile.replace("FRNet","PVKD")

    create(outfile)
    create(sp_outfile)
    create(pvkd_outfile)
    
    frnet.tofile(outfile)
    sphere.tofile(sp_outfile)
    pvkd.tofile(pvkd_outfile)
    # pdb.set_trace()
    err_fr = outfile.replace("predictions","ErrorSum").replace(".label",".npy")
    err_sp = sp_outfile.replace("predictions","ErrorSum").replace(".label",".npy")
    err_pvkd = pvkd_outfile.replace("predictions","ErrorSum").replace(".label",".npy")

    create(err_fr)
    create(err_sp)
    create(err_pvkd)
    fr_error_sum.tofile(err_fr)
    sp_error_sum.tofile(err_sp)
    pvkd_error_sum.tofile(err_pvkd)
    print(f'{err_fr}')
    # pdb.set_trace()
#
    # pdb.set_trace()

img /= len(im_file)
# pdb.set_trace()
    