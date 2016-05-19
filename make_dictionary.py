import numpy as np
import mini_batch_dictionary_learning as dl
import sys,os,json,imageio

def learn(params):
    imgs = map(lambda x: imageio.read_image_L(x), params['training_images'])
    n_imgs = len(imgs)
    n_samples = params['n_samples']
    pw, ph = params['pw'], params['ph']
    dl_params = dl.initialize_params(pw*ph, params['n_atoms'], params['sparsity'])
    for i in xrange(n_samples):
        idx_img = np.random.randint(n_imgs)
        h, w = imgs[idx_img].shape
        xn, yn = w-pw+1, h-ph+1
        x, y = np.random.randint(xn), np.random.randint(yn)
        print "learn ({}/{}) : {} [{}, {}]".format(i+1, n_samples, params['training_images'][idx_img], x, y)
        patch_img = imgs[idx_img][y:y+ph, x:x+pw]
        dl.learn(patch_img.ravel(), dl_params)
    return dl.get_dictionary(dl_params)

sys.argv.pop(0)
params = json.load(open(sys.argv[0]))
working_dir = os.path.join(params['working_dir'], params['setting_id'])
dict_path = os.path.join(working_dir, params['dictionary_filename'])

np.random.seed(0)
D = learn(params)

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

np.save(dict_path, D)
