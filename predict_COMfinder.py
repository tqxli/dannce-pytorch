# Scipt to run COM finding over a single experiment
# Usage: python ./predict_COMfinder path_to_config
import numpy as np
import scipy.io as sio
from copy import deepcopy

import sys
sys.path.append('./engine/')
sys.path.append('./utils')
sys.path.append('/usr/local/MATLAB/R2018a/extern/engines/python/build/lib.linux-x86_64-2.7')
import matlab.engine
import matlab
eng = matlab.engine.start_matlab()

import keras
from keras.losses import mean_squared_error

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import processing
import ops
from generator_aux import DataGenerator_downsample

from losses import mask_nan_keep_loss, mask_nan_keep_loss_safe, metric_dist_max
import nets

from six.moves import cPickle

import serve_data_COM as serve_data

params = processing.read_config(sys.argv[1])

params['loss'] = locals()[params['loss']]
params['net'] = eval(params['net'])

undistort = params['undistort']

# Build net
print("Initializing Network...")

model = params['net'](params['loss'],params['lr'],
							 params['N_CHANNELS_IN']
							 ,params['N_CHANNELS_OUT'],
							params['metric'],multigpu=False)
print("COMPLETE\n")

model.load_weights(params['weights'])

vid_dir_flag = params['vid_dir_flag']

def evaluate_COM_steps(start_ind,end_ind, steps):
	for i in range(start_ind,end_ind,steps):   
		# close videos from unneeded steps
		currentframes = labels[partition['valid'][i]]['frames']
		currentframes = min(list(currentframes.values()))
		
		m = min([i+steps,len(partition['valid'])-1])
		
		maxframes = labels[partition['valid'][m]]['frames']
		maxframes = min(list(maxframes.values()))
		lastvid = str(currentframes//3500*3500 - 3500) + '.mp4'
		for n in range(len(params['CAMNAMES'])):
			for key in list(vids[params['CAMNAMES'][n]].keys()):
				vikey = key.split('/')[1]
				if lastvid == vikey:
					print("Closing video: {}".format(key))
					vids[params['CAMNAMES'][n]][key].close()
	
		# Open new vids for this interval

		for j in range(len(params['CAMNAMES'])):
			if vid_dir_flag:
				addl = ''
			else:
				addl = os.listdir(os.path.join(params['viddir'],params['CAMNAMES'][j]))[0]

			vids[params['CAMNAMES'][j]] = processing.generate_readers(params['viddir'],
																			os.path.join(params['CAMNAMES'][j],addl),
																			minopt=currentframes//3500*3500,
																			maxopt=maxframes)
		
		valid_generator = DataGenerator_downsample(partition['valid'][i:i+steps], labels,vids, **valid_params)
		pred_ = model.predict_generator(valid_generator,
										steps=steps,
										verbose=1)
		
		print(i)
		pred_ = np.reshape(pred_,[steps, len(params['CAMNAMES']), pred_.shape[1],pred_.shape[2]])
		for m in range(len(partition['valid'][i:i+steps])): 
			#odd loop condition, but it's because at the end of samples, predict_generator will continue to
			#make predictions in a way I don't grasp yet, but also in a way we should ignore
			pred = pred_[m]
			sampleID_ = partition['valid'][i + m]
			save_data[sampleID_] = {}
			save_data[sampleID_]['triangulation'] = {}

			for j in range(pred.shape[0]): # this loops over all cameras
				# get coords for each map. This assumes that image are coming out in pred in the same order as
				# CONFIG_PARAMS['CAMNAMES']
				pred_max = np.max(np.squeeze(pred[j]))            
				ind = np.array(processing.get_peak_inds(np.squeeze(pred[j])))*2
				ind[0] += params['CROP_HEIGHT'][0]
				ind[1] += params['CROP_WIDTH'][0]
				ind = ind[::-1] # now, the center of mass is (x,y) instead of (i,j)
				# now, we need to use camera calibration to triangulate from 2D to 3D

				save_data[sampleID_][params['CAMNAMES'][j]] = {'pred_max': pred_max, 
									'COM': ind}

			#triangulate for all unique pairs
			for j in range(pred.shape[0]):
				for k in range(j+1,pred.shape[0]):
					pts1 = save_data[sampleID_][params['CAMNAMES'][j]]['COM']
					pts2 = save_data[sampleID_][params['CAMNAMES'][k]]['COM']
					pts1 = pts1[np.newaxis,:]
					pts2 = pts2[np.newaxis,:]
					
					if undistort:
						pts1 = ops.unDistortPoints(pts1,cameras[params['CAMNAMES'][j]]['K'],
							  cameras[params['CAMNAMES'][j]]['RDistort'],
							  cameras[params['CAMNAMES'][j]]['TDistort'])
						pts2 = ops.unDistortPoints(pts2,cameras[params['CAMNAMES'][k]]['K'],
							  cameras[params['CAMNAMES'][k]]['RDistort'],
							  cameras[params['CAMNAMES'][k]]['TDistort'])
					test3d = np.squeeze(ops.triangulate(pts1,pts2,camera_mats[params['CAMNAMES'][j]],
										 camera_mats[params['CAMNAMES'][k]]))
					save_data[sampleID_]['triangulation']["{}_{}".format(params['CAMNAMES'][j],
																 params['CAMNAMES'][k])] = test3d


RESULTSDIR = params['RESULTSDIR']
print(RESULTSDIR)

if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)

samples, datadict, datadict_3d, cameras, camera_mats, vids = serve_data.prepare_data(params,
																					vid_dir_flag=params['vid_dir_flag'],
																					minopt=0,
																					maxopt=0)

# Zero any negative frames -- DEPRECATED
for key in datadict.keys():
	for key_ in datadict[key]['frames'].keys():
		if datadict[key]['frames'][key_] < 0:
			datadict[key]['frames'][key_] = 0

# Parameters
valid_params = {'dim_in': (params['INPUT_HEIGHT'],params['INPUT_WIDTH']),
		  'n_channels_in': params['N_CHANNELS_IN'],
		  'dim_out': (params['OUTPUT_HEIGHT'],params['OUTPUT_WIDTH']),
		  'batch_size': params['BATCH_SIZE'],
		  'n_channels_out': params['N_CHANNELS_OUT'],
		  'out_scale': params['SIGMA'],
		  'camnames': params['CAMNAMES'],
		  'crop_width': params['CROP_WIDTH'],
		  'crop_height': params['CROP_HEIGHT'],
		  'bbox_dim': (params['BBOX_HEIGHT'],params['BBOX_WIDTH']),
		  'downsample': params['DOWNFAC'],
		  'labelmode': params['LABELMODE'],      
		  'shuffle': False}

partition = {}
partition['valid'] = samples
labels = datadict
labels_3d = datadict_3d

save_data = {}

evaluate_COM_steps(0,len(samples),3500)

#Close video objects
for j in range(len(params['CAMNAMES'])):
	for key in vids[params['CAMNAMES'][j]]:
		vids[params['CAMNAMES'][j]][key].close()

if undistort == True:
	# Then undistortion already happened, just save
	f = open(RESULTSDIR + 'COM_undistorted.pickle','wb')
	cPickle.dump(save_data,f)
	f.close()

else:
	# We need to (awkwardly) send our data into matlab for bulk undistortion

	# Colate all coms
	n  = len(list(save_data.keys()))
	num_cams = len(params['CAMNAMES'])
	allCOMs = np.zeros((num_cams,n,2))
	for (i,  key) in enumerate(save_data.keys()):
		for c in range(num_cams):
			allCOMs[c,i] = save_data[key][params['CAMNAMES'][c]]['COM']

	comfile = 'allCOMs_distorted.mat'
	sio.savemat(comfile,{'allCOMs':allCOMs})

	eng.undistort_allCOMS(comfile,params['CALIB_DIR'] + 'worldcoordinates_lframe.mat',matlab.double([2,3,1]),nargout=0)

	allCOMs_u = sio.loadmat('allCOMs_undistorted.mat')['allCOMs_u']

	os.remove('allCOMs_distorted.mat')
	os.remove('allCOMs_undistorted.mat')

	save_data_u = deepcopy(save_data)
	num_cams = len(params['CAMNAMES'])
	for (i,  key) in enumerate(save_data.keys()):
	    for c in range(num_cams):
	        save_data_u[key][params['CAMNAMES'][c]]['COM'] = allCOMs_u[c,i] 

	f = open(RESULTSDIR + 'COM_undistorted.pickle','wb')
	cPickle.dump(save_data_u,f)
	f.close()

print('done!')

