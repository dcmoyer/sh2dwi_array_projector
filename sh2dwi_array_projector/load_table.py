
import numpy as np
import tensorflow as tf


class BMatTable():

  ##
  ## Input:
  ##  dictionary of keys and filenames, or keys and arrays
  ##
  def __init__( keys_and_mats ):

    for key, value in keys_and_mats.items():
      if value is str:
        #read in the matrix
        print("haven't implemented the read in bits yet, just do it yerself!")
        exit(1)

    self.b_mat_lookup_table = keys_and_mats

  ##
  ##  This function makes a short function to produce onehot vectors with
  ##  a specified set of keys. It raises a KeyError whenever it encounters
  ##  a key not in that set. It's meant to be used in a TF preprocessing
  ##  function.
  def get_subset( list_of_scan_keys, just_the_stack=False):

    # check if all keys are in list_of_s
    for key in list_of_scan_keys:
      if key not in self.b_mat_lookup_table.keys():
        raise KeyError("key not found in this BMatTable")
      #if list_of_scan_keys.count(key) > 1:
      #  raise KeyError("key duplicated in input list")

    #construct a bmat block
    #TODO: here's where the padding should go, so that the largest output sets
    # the block size. Also, we need to put weight vectors for each of the
    # outputs, so that zeros aren't counted in MSE
    list_of_bmats = [ \
      self.b_mat_lookup_table[key] for key in list_of_scan_keys
    ]
    b_mat_tensor = tf.convert_to_tensor( \
      np.array(list_of_bmats),
      dtype=tf.float32
    ) 

    if just_the_stack:
      return b_mat_tensor

    def get_one_hot_func(key):
      if key not in list_of_scan_keys:
        raise KeyError("bad key passed to selection vector creation")
      return tf.one_hot( \
        list_of_scan_keys.index(key), len(list_of_scan_keys)
      )

    return get_one_hot_func, b_mat_tensor


##
## This function takes b_mat_tensor stacks and indicator vectors to specific
##  stacks. This prevents having to re-roll the b_mat_tensor every batch,
##  instead just re-rolling the one hot function.
##
def b_mat_tensor_to_subj(name, B_stack, sel_vec):

  with tf.variable_scope(("%s_tensor_mult_out" % name), reuse=tf.AUTO_REUSE):
    #print("[sh2dwi_array_projector] b_stack shape ", B_stack.shape)
    #print("[sh2dwi_array_projector] sel_vec shape ", sel_vec.shape)
    B = tf.tensordot(B_stack,sel_vec,axes=[[0],[1]])
    B = tf.transpose(B,perm=[2,0,1])
    #print(B.shape)

  return B













