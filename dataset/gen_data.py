import numpy as np
import cv2 as cv
import argparse

# import the environment and tester registration
import gym_control_model.planararm.register_env
import gym

__author__ = "Sayantan Auddy"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Development"

"""
Code for generating robot arm images from observations
"""

gym.logger.set_level(40)

def generate_save_images(dataset_file, img_cnt, img_size, img_binary, dtype_num):

    # Observations are images
    env = gym.make("PlanarArmTeacher2Learner3-v2")

    # Checking img_size
    learner_img, _ = env.reset()

    if dtype_num==16:
        dtype_np = np.float16
    elif dtype_num==32:
        dtype_np = np.float32
    elif dtype_num==64:
        dtype_np = np.float64

    image_dataset = np.zeros((img_cnt, img_size, img_size), dtype=dtype_np)

    
    for i in range(img_cnt//2):
        # Get observation images
        learner_img, teacher_img = env.reset()

        # Scale the images
        learner_img = cv.resize(learner_img, (img_size,img_size), interpolation=cv.INTER_AREA)
        teacher_img = cv.resize(teacher_img, (img_size,img_size), interpolation=cv.INTER_AREA)

        # Convert to binary images
        learner_img = cv.cvtColor(learner_img, cv.COLOR_BGR2GRAY)
        if img_binary:
            _, learner_img = cv.threshold(learner_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        teacher_img = cv.cvtColor(teacher_img, cv.COLOR_BGR2GRAY)
        if img_binary:
            _, teacher_img = cv.threshold(teacher_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        # Convert the pixels from 0..255 to 0..1
        learner_img = learner_img.astype(np.float)/255.0
        teacher_img = teacher_img.astype(np.float)/255.0

        image_dataset[2*i] = learner_img
        image_dataset[2*i+1] = teacher_img

    np.savez_compressed(dataset_file, array=image_dataset)

def generate_images_agent(observation, img_size, img_binary, dtype_num):
    
    if dtype_num==16:
        dtype_np = np.float16
    elif dtype_num==32:
        dtype_np = np.float32
    elif dtype_num==64:
        dtype_np = np.float64
        
    # Scale the images
    learner_img = cv.resize(observation[0], (img_size,img_size), interpolation=cv.INTER_AREA)
    teacher_img = cv.resize(observation[1], (img_size,img_size), interpolation=cv.INTER_AREA)

    # Convert to binary images
    learner_img = cv.cvtColor(learner_img, cv.COLOR_BGR2GRAY)
    if img_binary:
        _, learner_img = cv.threshold(learner_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    teacher_img = cv.cvtColor(teacher_img, cv.COLOR_BGR2GRAY)
    if img_binary:
        _, teacher_img = cv.threshold(teacher_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Convert the pixels from 0..255 to 0..1
    learner_img = learner_img.astype(np.float)/255.0
    teacher_img = teacher_img.astype(np.float)/255.0
    
    return learner_img, teacher_img


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate state images from the environment')
    parser.add_argument('--f', dest='dataset_file', type=str, required=True, help='File name for image array')
    parser.add_argument('--c', dest='img_cnt', type=int, required=False, default=10000, help='Total count of images (must be an even number)')
    parser.add_argument('--s', dest='img_size', type=int, required=False, default=128, help='Size of an image')
    parser.add_argument('--b', dest='img_binary', default=False, action='store_true', help='Provide this flag to create binary images')
    parser.add_argument('--d', dest='dtype_num', type=int, default=64, help='dtype for image array')

    args = parser.parse_args()

    assert args.img_cnt%2==0, "img_cnt needs to be an even number"
    assert args.img_size!=0, "img_size cannot be 0"

    generate_save_images(args.dataset_file, args.img_cnt, args.img_size, args.img_binary, args.dtype_num)
    





