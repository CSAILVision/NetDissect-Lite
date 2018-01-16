
import os
from torch.autograd import Variable as V
from scipy.misc import imresize
import numpy as np
import torch
import settings
from data_loader.loadseg import SegmentationData, SegmentationPrefetcher

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


class FeatureOperator:

    def __init__(self):
        if not os.path.exists(settings.OUTPUT_FOLDER):
            os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'image'))
        self.data = SegmentationData(settings.DATA_DIRECTORY)
        self.loader = SegmentationPrefetcher(self.data,categories=['image'],once=True,batch_size=settings.BATCH_SIZE)
        self.mean = [109.5388,118.6897,124.6901]

    def feature_extraction(self,model=None,memmap=True):
        loader = self.loader
        # extract the max value activaiton for each image
        imglist_results = []
        maxfeatures = [None] * len(settings.FEATURE_NAMES)
        wholefeatures = [None] * len(settings.FEATURE_NAMES)
        features_size = [None] * len(settings.FEATURE_NAMES)
        features_size_file = os.path.join(settings.OUTPUT_FOLDER, "feature_size.npy")

        if memmap:
            skip = True
            mmap_files =  [os.path.join(settings.OUTPUT_FOLDER, "%s.mmap" % feature_name)  for feature_name in  settings.FEATURE_NAMES]
            mmap_max_files = [os.path.join(settings.OUTPUT_FOLDER, "%s_max.mmap" % feature_name) for feature_name in settings.FEATURE_NAMES]
            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file)
            else:
                skip = False
            for i, (mmap_file, mmap_max_file) in enumerate(zip(mmap_files,mmap_max_files)):
                if os.path.exists(mmap_file) and os.path.exists(mmap_max_file) and features_size[i] is not None:
                    print('loading features %s' % settings.FEATURE_NAMES[i])
                    wholefeatures[i] = np.memmap(mmap_file, dtype=float,mode='r', shape=tuple(features_size[i]))
                    maxfeatures[i] = np.memmap(mmap_max_file, dtype=float, mode='r', shape=tuple(features_size[i][:2]))
                else:
                    print('file missing, load again')
                    skip = False
            if skip:
                return wholefeatures, maxfeatures

        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size
        for batch_idx,batch in enumerate(loader.tensor_batches(bgr_mean=self.mean)):
            del features_blobs[:]
            input = batch[0]
            batch_size = len(input)
            print('extracting feature from batch %d / %d' % (batch_idx+1, num_batches))
            input = torch.from_numpy(input[:, ::-1, :, :].copy())
            input.div_(255.0 * 0.224)
            if settings.GPU:
                input = input.cuda()
            input_var = V(input,volatile=True)
            logit = model.forward(input_var)
            while np.isnan(logit.data.max()):
                print("nan") #which I have no idea why it will happen
                del features_blobs[:]
                logit = model.forward(input_var)
            if maxfeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (len(loader.indexes), feat_batch.shape[1])
                    if memmap:
                        maxfeatures[i] = np.memmap(mmap_max_files[i],dtype=float,mode='w+',shape=size_features)
                    else:
                        maxfeatures[i] = np.zeros(size_features)
            if wholefeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (len(loader.indexes), feat_batch.shape[1], feat_batch.shape[2], feat_batch.shape[3])
                    features_size[i] = size_features
                    if memmap:
                        wholefeatures[i] = np.memmap(mmap_files[i],dtype=float,mode='w+',shape=size_features)
                    else:
                        wholefeatures[i] = np.zeros(size_features)
            np.save(features_size_file,features_size)
            start_idx = batch_idx*settings.BATCH_SIZE
            end_idx = min((batch_idx+1)*settings.BATCH_SIZE, len(loader.indexes))
            for i, feat_batch in enumerate(features_blobs):
                wholefeatures[i][start_idx:end_idx] = feat_batch
                if len(feat_batch.shape) == 4:
                    maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch,3),2)
                elif len(feat_batch.shape) == 3:
                    maxfeatures[i][start_idx:end_idx] = np.max(feat_batch, 2)
                elif len(feat_batch.shape) == 2:
                    maxfeatures[i][start_idx:end_idx] = feat_batch
        return wholefeatures,maxfeatures

    def quantile_threshold(self, features):
        print("calculating quantile threshold")
        if len(features.shape) == 4:
            axis=[0,2,3]
        elif len(features.shape) == 2:
            axis=[0]
        return np.percentile(features,100*(1 - settings.QUANTILE),axis=axis)

    def tally(self, features, threshold, parallel=0):
        data  = self.data
        units = features.shape[1]
        labels = len(data.label)
        tally_both = np.zeros((units,labels),dtype=np.uint64)
        tally_units = np.zeros(units,dtype=np.uint64)
        tally_labels = np.zeros(labels,dtype=np.uint64)
        pd = SegmentationPrefetcher(data,categories=data.category_names(),
                                    once=True,batch_size=settings.BATCH_SIZE,
                                    ahead=settings.BATCH_SIZE)

        for batch in pd.batches():
            for concept_map in batch:
                img_index = concept_map['i']
                print('labelprobe image index %s' % str(img_index))
                scalars,pixels = [],[]
                for cat in data.category_names():
                    label_group = concept_map[cat]
                    shape = np.shape(label_group)
                    if len(shape) % 2 == 0:
                        label_group = [label_group]
                    if len(shape) < 2:
                        scalars += label_group
                    else:
                        pixels.append(label_group)
                for scalar in scalars:
                    tally_labels[scalar] += concept_map['sh'] * concept_map['sw']
                for pixel in pixels:
                    for si in range(concept_map['sh']):
                        for sj in range(concept_map['sw']):
                            tally_labels[pixel[0,si,sj]] += 1

                for unit_id in range(units):
                    feature_map = features[img_index][unit_id]
                    if feature_map.max() > threshold[unit_id]:
                        mask = imresize(feature_map, (concept_map['sh'],concept_map['sw']), mode='F')
                        indexes = np.argwhere(mask > threshold[unit_id])
                        tally_units[unit_id] += len(indexes)
                        for pixel in pixels:
                            for index in indexes:
                                tally_both[unit_id, pixel[0,index[0],index[1]]] += 1
                        for scalar in scalars:
                            tally_both[unit_id, scalar] += len(indexes)

        categories = data.category_names()
        primary_categories = primary_categories_per_index(data, categories=categories)
        labelcat = onehot(primary_categories)
        iou = tally_both / (tally_units.transpose()[:,np.newaxis] + tally_labels[np.newaxis,:] - tally_both + 1e-10)
        pciou = np.array([iou * (primary_categories[np.arange(iou.shape[1])] == ci)[np.newaxis, :] for ci in range(len(data.category_names()))])
        label_pciou = pciou.argmax(axis=2)
        name_pciou = [
            [data.name(None, j) for j in label_pciou[ci]]
            for ci in range(len(label_pciou))]
        score_pciou = pciou[
            np.arange(pciou.shape[0])[:, np.newaxis],
            np.arange(pciou.shape[1])[np.newaxis, :],
            label_pciou]
        bestcat_pciou = score_pciou.argsort(axis=0)[::-1]
        ordering = score_pciou.max(axis=0).argsort()[::-1]
        rets = [None] * len(ordering)
        for i,unit in enumerate(ordering):
            # Top images are top[unit]
            bestcat = bestcat_pciou[0, unit]
            data = {
                'unit': (unit + 1),
                'category': categories[bestcat],
                'label': name_pciou[bestcat][unit],
                'score': score_pciou[bestcat][unit]
            }
            for ci, cat in enumerate(categories):
                label = label_pciou[ci][unit]
                data.update({
                    '%s-label' % cat: name_pciou[ci][unit],
                    '%s-truth' % cat: tally_labels[label],
                    '%s-activation' % cat: tally_units[unit],
                    '%s-intersect' % cat: tally_both[unit, label],
                    '%s-iou' % cat: score_pciou[ci][unit]
                })
            rets[i] = data
        return rets

def primary_categories_per_index(ds, categories=None):
    '''
    Returns an array of primary category numbers for each label, where the
    first category listed in ds.category_names is given category number 0.
    '''
    catmap = {}
    for cat in categories:
        imap = ds.category_index_map(cat)
        if len(imap) < ds.label_size(None):
            imap = np.concatenate((imap, np.zeros(
                ds.label_size(None) - len(imap), dtype=imap.dtype)))
        catmap[cat] = imap
    result = []
    for i in range(ds.label_size(None)):
        maxcov, maxcat = max(
            (ds.coverage(cat, catmap[cat][i]) if catmap[cat][i] else 0, ic)
            for ic, cat in enumerate(categories))
        result.append(maxcat)
    return np.array(result)

def onehot(arr, minlength=None):
    '''
    Expands an array of integers in one-hot encoding by adding a new last
    dimension, leaving zeros everywhere except for the nth dimension, where
    the original array contained the integer n.  The minlength parameter is
    used to indcate the minimum size of the new dimension.
    '''
    length = np.amax(arr) + 1
    if minlength is not None:
        length = max(minlength, length)
    result = np.zeros(arr.shape + (length,))
    result[list(np.indices(arr.shape)) + [arr]] = 1
    return result