import settings
from data_loader.model_loader import loadmodel
from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary

fo = FeatureOperator()
model = loadmodel(hook_feature)

############ STEP 1: feature extraction ###############
features, maxfeature = fo.feature_extraction(model=model, feature_names=settings.FEATURE_NAMES)

for layer_id,layer in enumerate(settings.FEATURE_NAMES):
############ STEP 2: calculating threshold ############
    thresholds = fo.quantile_threshold(maxfeature[layer_id])

############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features[layer_id],thresholds)

############ STEP 4: generating results ###############
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
