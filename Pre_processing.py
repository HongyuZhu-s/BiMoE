import _pickle as cPickle
import os
import numpy as np
import os.path as osp
import scipy.io as sio
import numpy as np
import torch
import h5py

class PrepareData:
    def __init__(self, args):

        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                               'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                               'CP2', 'P4', 'P8', 'PO4', 'O2']


    def run(self, subject_list, split=False, feature=False):

        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)
            label_ = self.label_selection(label_)

            if split:
                data_, label_ = self.split(data=data_, label=label_, segment_length=self.args.segment,
                                           overlap=self.args.overlap, sampling_rate=self.args.sampling_rate)

            self.save(data_, label_, sub)


    def load_data_per_subject(self, sub):

        sub += 1
        if (sub < 10):
            sub_code = str('s0' + str(sub) + '.dat')
        else:
            sub_code = str('s' + str(sub) + '.dat')

        subject_path = os.path.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['labels']
        data = subject['data'][:, :, 3 * 128:]

        return data, label

    def label_selection(self, label):

        if self.label_type == 'A':
            label = label[:, 1]
        elif self.label_type == 'V':
            label = label[:, 0]
        elif self.label_type == 'D':
            label = label[:, 2]
        elif self.label_type == 'L':
            label = label[:, 3]
        if self.args.num_class == 2:
            label = np.where(label <= 5, 0, label)
            label = np.where(label > 5, 1, label)

        return label

    def split(self, data, label, segment_length=1, overlap=0, sampling_rate=256):
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[2] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        data = data_split_array
        assert len(data) == len(label)
        return data, label

    def save(self, data, label, sub):
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def load_all_subjects(data_folder, subject_count=32):
        all_data = []
        all_labels = []
        subject_info = []

        for sub_id in range(subject_count):
            file_path = os.path.join(data_folder, f'sub{sub_id}.hdf')

            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping this test subject.")
                continue

            with h5py.File(file_path, 'r') as f:
                data = f['data'][:]
                labels = f['label'][:]

            data_tensor = torch.from_numpy(data).float()
            labels_tensor = torch.from_numpy(labels).long()

            all_data.append(data_tensor)
            all_labels.append(labels_tensor)

            subject_info.append({
                'subject_id': sub_id,
                'data_shape': data_tensor.shape,
                'label_shape': labels_tensor.shape
            })

            print(f"Subject {sub_id}: data shape {data_tensor.shape}, label shape {labels_tensor.shape}")

        return all_data, all_labels, subject_info