
import numpy as np
import tensorflow as tf
from collections import defaultdict

class BMatTable():

  ##
  ## Input:
  ##  dictionary of keys and filenames, or keys and arrays
  ##
  def __init__( self, keys_and_mats, keys_and_weights=None):

    for key, value in keys_and_mats.items():
      if isinstance(value,str):
        #read in the matrix
        if value.endswith(".npy"):
          keys_and_mats[key] = np.load(value)
        else:
          print("other filetypes not implemented, just do it yerself!")
          exit(1)

    #self.b_mat_lookup_table = keys_and_mats
    #self.b_mat_lookup_table = \
    #  tf.contrib.lookup.HashTable(\
    #    tf.contrib.lookup.KeyValueTensorInitializer( \
    #      tf.convert_to_tensor(list(keys_and_mats.keys())), \
    #      tf.convert_to_tensor(list(keys_and_mats.values())), \
    #    ),\
    #    np.nan
    #  )

    self.list_of_keys = list(keys_and_mats.keys())
    self.list_of_tensors = [keys_and_mats[key] for key in self.list_of_keys]
    self.b_mat_tensor = tf.convert_to_tensor( \
      np.array(self.list_of_tensors),
      dtype=tf.float32
    ) 

    if not keys_and_weights:
      self.weight_lookup_table = None
      return

    for key, value in keys_and_weights.items():
      if key not in self.list_of_keys:
        print("key in weights not found in mats dict")
        exit(1)
      if isinstance(value,str):
        #read in the matrix
        if value.endswith(".npy"):
          keys_and_weights[key] = np.load(value)
        else:
          print("other filetypes not implemented, just do it yerself!")
          exit(1)
    #self.weight_lookup_table = keys_and_weights
    #self.weight_lookup_table = \
    #  tf.contrib.lookup.HashTable(\
    #    tf.contrib.lookup.KeyValueTensorInitializer( \
    #      tf.convert_to_tensor(list(keys_and_weights.keys())), \
    #      tf.convert_to_tensor(list(keys_and_weights.values())), \
    #    ),\
    #    np.nan
    #  )
    self.list_of_weights = [keys_and_weights[key] for key in self.list_of_keys]
    self.weight_tensor = tf.convert_to_tensor( \
      np.array(self.list_of_weights),
      dtype=tf.float32
    ) 

  ##
  ##  This function makes a short function to produce onehot vectors with
  ##  a specified set of keys. It raises a KeyError whenever it encounters
  ##  a key not in that set. It's meant to be used in a TF preprocessing
  ##  function.
  ## BUT if you can hold the stack in memory, just...do that instead?
  def get_subset( self, list_of_scan_keys, just_the_stack=False):

    # check if all keys are in list_of_s
    for key in list_of_scan_keys:
      if key not in self.list_of_keys:
        raise KeyError("key not found in this BMatTable")
      #if list_of_scan_keys.count(key) > 1:
      #  raise KeyError("key duplicated in input list")

    #construct a bmat block
    #TODO: here's where the padding should go, so that the largest output sets
    # the block size. Also, we need to put weight vectors for each of the
    # outputs, so that zeros aren't counted in MSE
    #list_of_bmats = [ \
    #  self.b_mat_lookup_table[key] for key in list_of_scan_keys
    #]
    #b_mat_tensor = tf.convert_to_tensor( \
    #  np.array(list_of_bmats),
    #  dtype=tf.float32
    #) 

    index_list = [ self.list_of_keys.index(key) for key in self.list_of_keys]
    self.one_hot_lookup = tf.contrib.lookup.HashTable(\
        tf.contrib.lookup.KeyValueTensorInitializer( \
          tf.convert_to_tensor(self.list_of_keys), \
          tf.convert_to_tensor(index_list), \
        ),\
        -1919
      )

    if just_the_stack:
      return self.b_mat_tensor

    def get_one_hot_func(key):
      #if key not in self.list_of_keys:
      #  print(key)
      #  raise KeyError("bad key passed to selection vector creation")
      #return tf.one_hot( self.list_of_keys.index(key), len(list_of_keys) )
      return tf.squeeze(tf.one_hot( self.one_hot_lookup.lookup(key), len(self.list_of_keys) ))

    #def get_weights_func(key):
    #  if self.weight_lookup_table is None:
    #    return 1.0
    #  #elif key not in list_of_scan_keys:
    #  #  raise KeyError("bad key passed to selection vector creation")
    #  #return self.weight_lookup_table[ key ]
    #  return self.weight_lookup_table.lookup( key )

    return get_one_hot_func, self.b_mat_tensor, self.weight_tensor, self.one_hot_lookup.init

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

##
## This function takes b_mat_tensor stacks and indicator vectors to specific
##  stacks. This prevents having to re-roll the b_mat_tensor every batch,
##  instead just re-rolling the one hot function.
##
def weight_tensor_to_subj(name, weight_tensor, sel_vec):

  with tf.variable_scope(("%s_weight_tensor_mult_out" % name), reuse=tf.AUTO_REUSE):
    #print("[sh2dwi_array_projector] b_stack shape ", B_stack.shape)
    #print("[sh2dwi_array_projector] sel_vec shape ", sel_vec.shape)
    weights = tf.matmul(sel_vec,weight_tensor)
    #B = tf.transpose(B,perm=[2,0,1])
    #print(B.shape)

  return weights













