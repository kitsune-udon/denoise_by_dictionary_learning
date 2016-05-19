import numpy as np
import sys,os,json,imageio

def visualize_dictionary(out_path, dictionary, patch_size):
    n_atoms = dictionary.shape[1]
    pw, ph = patch_size
    bxn = 10
    byn = ((n_atoms-1)/bxn)+1
    r = np.zeros((ph*byn, pw*bxn))
    scaler = 0.5 / np.max(np.fabs(dictionary))
    for i in xrange(n_atoms):
        bx, by = i % bxn, i / bxn
        x, y = bx * pw, by * ph
        r[y:y+ph, x:x+pw] = dictionary[:, i].reshape((ph, pw))
    imageio.write_image_L(out_path, scaler * (r+0.5))

sys.argv.pop(0)
params = json.load(open(sys.argv[0]))

patch_size = (params['pw'], params['ph'])
working_dir = os.path.join(params['working_dir'], params['setting_id'])
dict_path = os.path.join(working_dir, params['dictionary_filename'])
dict_img_path = os.path.join(working_dir, params['dictionary_img_filename'])
dictionary = np.load(dict_path)
visualize_dictionary(dict_img_path, dictionary, patch_size)
