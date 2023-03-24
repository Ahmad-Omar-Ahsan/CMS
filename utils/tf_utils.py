import tensorflow as tf
import numpy as np
import os

def load_dataset(config : dict):
    data_dir = config['data_dir']
    dataset = np.load(os.path.join(data_dir, "dataset.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))

    return create_tf_dataset(dataset_np=dataset,labels_np=labels, batch_size=config['hparams']['batch_size'])


def create_tf_dataset(dataset_np, labels_np, batch_size):
    tf_dataset = tf.data.Dataset.from_tensor_slices((dataset_np, labels_np))
    length_ds = len(dataset_np)
    train_size = int(0.8 * length_ds)
    val_size = int(0.1 * length_ds)
    test_size = length_ds - train_size - val_size



    train_dataset = tf_dataset.take(train_size)
    test_dataset = tf_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset


if __name__=="__main__":
    config = {
        "data_dir": "/media/saitomar/Work/Projects/CMS/np_dataset"
    }
    train_dataset, val_dataset, test_dataset = load_dataset(config=config)
    print(type(train_dataset), len(train_dataset), len(val_dataset), len(test_dataset))
