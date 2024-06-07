import argparse
import traceback

import numpy as np
import tensorflow as tf
import pathlib

# https://www.appsloveworld.com/tensorflow/10/tensorboard-export-csv-file-from-command-line
def save_tag_to_csv(fn, tag='test_metric'):
    parts = fn.parts
    parent = pathlib.Path(*parts[0:-2])
    subfolder_name = parts[-2]

    output_fn = '{}/{}_{}.csv'.format(parent, subfolder_name, tag.replace('/', '_'))
    # print("Will save to {}".format(output_fn))

    # sess = tf.InteractiveSession()

    wall_step_values = []
    # with sess.as_default():
    for e in tf.compat.v1.train.summary_iterator(str(fn)):
        for v in e.summary.value:
            if v.tag == tag:
                wall_step_values.append((e.wall_time, e.step, v.simple_value))
    np.savetxt(output_fn, wall_step_values, delimiter=',', fmt='%10.5f', header="wall_time,step,{}".format(tag), comments='') # If I don't set comments, the header has a hash tag in front

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=".")
    parser.add_argument('--tag', default="experience_generation/episode_rew")
    args = parser.parse_args()

    root = pathlib.Path(args.path)
    filenames = root.rglob("event*")


    for filename in filenames:
        # print(f"{filename}\n\n")
        try:
            # print("Processing: {}".format(filename))
            save_tag_to_csv(filename, tag=args.tag)
        except Exception:
            print(traceback.format_exc())
