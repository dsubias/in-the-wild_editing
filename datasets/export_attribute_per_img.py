import os.path
import argparse
import numpy as np
import random
import glob

def parse_args(required=True):
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Train semantic boundary with given latent codes and '
                    'attribute scores.')
    parser.add_argument('-i', '--image_path', type=str, required=True, default="../dataset/renders_by_geom_ldr/", 
                        help='Path to the input latent codes. (required)')
    parser.add_argument('-s', '--scores_path', type=str, required=False, default="all_attribute_score.npy", 
                        help='Path to the input attribute scores. (required)')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='train/test...')  
    return parser.parse_args()

def add_image_to_dataset(im,all_scores,attributes,outfile, has_gt_attr):
    im_name=im.split("/")[-1]
    dset=im.split("/")[-2]
    #get the name of the material + write
    name = im_name.split('@')[-2].replace("-","")
    name = name.replace("_","")
#    print(im_name,name)
#    print(im,name)
    if(name in all_scores['material_name']):
        outfile.write(dset+"/"+im_name+"\t")
        #get the scores + write
        index=all_scores['material_name'].index(name)
        for att in attributes:
            outfile.write(str(float(all_scores[att][index])))
            outfile.write('\t')
        outfile.write('\n')
    elif(not has_gt_attr):
        outfile.write(dset+"/"+im_name+"\t")
        #get the scores + write
        for att in attributes:
            outfile.write(str(0.5))
            outfile.write('\t')
        outfile.write('\n')

def get_random_mat(mat_names,ratio):
    n_mat=int(len(mat_names)*ratio/100)
    print("Selecting randomly %i materials over %i"%(n_mat,len(mat_names)))
    selected_mat=random.sample(mat_names, n_mat)
    print(selected_mat)
    return selected_mat




def main():
    """Main function."""
    args = parse_args(False)
    #load the scores
    all_scores = np.load(args.scores_path, allow_pickle=True).item()
    attributes = list(all_scores.keys())[:-1]
    print(all_scores['material_name'])
    print(attributes)
    #create the output file and write attribute names
    outfile=open(args.image_path+"/attributes_dataset_{}.txt".format(args.dataset),'w')
    for att in attributes:
        outfile.write(att+"\t")
    outfile.write("\n")
    #get the list of images
    #print(os.path.join(args.image_path,"renderings",args.dataset,"*"))
    images = np.sort(glob.glob(os.path.join(args.image_path,"renderings",args.dataset,"*")))
    #print(images)
    print(len(images))
    #use gt attributes only if train dataset
    for im in images:
        add_image_to_dataset(im,all_scores,attributes,outfile,"train" in args.dataset)

main()
