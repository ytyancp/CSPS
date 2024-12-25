# -*- coding: utf-8 -*-
# @Time : 2024/10/1 14:50
# @Author : WeiqingWang
# @File : CSPHS_method
# @IDE : PyCharm 2023.1.5
# @Python : Python 3.10.7(64-bit)
# @Mail : wqwangahu@@163.com
# @Explain :CSPHS Algorithm

import numpy
import pandas
import collections
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


class CSPHS(object):
    # Initialization
    def __init__(self):
        self.data = None  # Dataset
        self.min_label = None   # Minority Label
        self.maj_label = None   # Majority Label
        self.min_data = None  # Minority Set
        self.maj_data = None  # Majority Set
        self.dimension = None  # Dataset Dimension
        self.imbalance_rate = None  # IR
        self.csp_center = pandas.DataFrame()  # CSP centers
        self.csp_data = pandas.DataFrame()  # CSP data
        self.hybrid_subset = pandas.DataFrame()  # Hybrid Subset
        self.seed = pandas.DataFrame()  # Seed
        self.auxiliary = dict()  # Auxiliary
        self.synthesis = None  # synthesis
        self.result = None  # Result

    # Constructive Sample Partition
    def constructiveSamplePartition(self):
        # Preprocessing
        data_pre = self.data.copy()
        # Calculate constraint radius R0
        radius_0 = 0
        for row_index, row in data_pre.iterrows():
            temp_other_data = data_pre.copy()
            temp_other_data['distance'] = 0
            for dimension in self.dimension:
                temp_other_data['distance'] = temp_other_data['distance'] + (temp_other_data[dimension] - row[dimension]) ** 2
            temp_other_data['distance'] = numpy.sqrt(temp_other_data['distance'])
            radius_0 = radius_0 + temp_other_data['distance'].sum()
        radius_0 = radius_0 / (data_pre.shape[0] * (data_pre.shape[0] - 1))

        # Construct the first partition subset
        subset_id = 1
        # Find zoning anchor
        gravity = data_pre.mean()
        data_pre['distance'] = 0
        for dimension in self.dimension:
            data_pre['distance'] = data_pre['distance'] + (data_pre[dimension] - gravity[dimension]) ** 2
        data_pre['distance'] = numpy.sqrt(data_pre['distance'])
        temp_center = data_pre.loc[data_pre['distance'].idxmin()].copy()
        center = temp_center.to_frame()
        center = pandas.DataFrame(center.values.T, columns=center.index, index=center.columns)
        # Find zoning radius
        data_pre = data_pre.drop(index=center.index)
        data_pre['distance'] = 0
        for dimension in self.dimension:
            data_pre['distance'] = data_pre['distance'] + (data_pre[dimension] - temp_center[dimension]) ** 2
        data_pre['distance'] = numpy.sqrt(data_pre['distance'])
        radius = data_pre['distance'].mean()
        radius = radius_0 if radius >= radius_0 else radius
        # Construct the subset
        subset_index = data_pre.loc[data_pre['distance'] <= radius].index
        subset = data_pre.loc[subset_index].copy()
        data_pre = pandas.concat([data_pre, center], axis=0)
        subset = pandas.concat([subset, center], axis=0)

        # Optimization the subset
        # Optimization zoning anchor
        gravity = subset.mean()
        subset['distance'] = 0
        for dimension in self.dimension:
            subset['distance'] = subset['distance'] + (subset[dimension] - gravity[dimension]) ** 2
        subset['distance'] = numpy.sqrt(subset['distance'])
        temp_center = subset.loc[subset['distance'].idxmin()].copy()
        center = temp_center.to_frame()
        center = pandas.DataFrame(center.values.T, columns=center.index, index=center.columns)
        # Optimization zoning radius
        subset = subset.drop(index=center.index)
        subset['distance'] = 0
        for dimension in self.dimension:
            subset['distance'] = subset['distance'] + (subset[dimension] - temp_center[dimension]) ** 2
        subset['distance'] = numpy.sqrt(subset['distance'])
        radius = subset['distance'].mean()
        radius = 0 if radius is numpy.NaN else radius
        # Optimization the subset
        data_pre['distance'] = 0
        for dimension in self.dimension:
            data_pre['distance'] = data_pre['distance'] + (data_pre[dimension] - temp_center[dimension]) ** 2
        data_pre['distance'] = numpy.sqrt(data_pre['distance'])
        subset_index = data_pre.loc[data_pre['distance'] <= radius].index
        subset = data_pre.loc[subset_index].copy()
        # Updating data
        self.csp_center = pandas.concat([self.csp_center, center], axis=0)
        self.csp_center.loc[center.index, 'radius'] = radius
        self.csp_center.loc[center.index, 'amount'] = int(subset.shape[0])
        self.csp_center.loc[center.index, 'subset_id'] = subset_id
        subset['subset_id'] = subset_id
        data_pre = data_pre.drop(index=subset_index, inplace=False)
        self.csp_data = pandas.concat([self.csp_data, subset], axis=0)

        # Construct the subsequent partition subset
        while data_pre.shape[0] > 0:
            subset_id = subset_id + 1
            center_last = temp_center
            # Find zoning anchor
            data_pre['distance'] = 0
            for dimension in self.dimension:
                data_pre['distance'] = data_pre['distance'] + (data_pre[dimension] - center_last[dimension]) ** 2
            data_pre['distance'] = numpy.sqrt(data_pre['distance'])
            temp_center = data_pre.loc[data_pre['distance'].idxmax()].copy()
            center = temp_center.to_frame()
            center = pandas.DataFrame(center.values.T, columns=center.index, index=center.columns)
            # Find zoning radius
            data_pre = data_pre.drop(index=center.index)
            data_pre['distance'] = 0
            for dimension in self.dimension:
                data_pre['distance'] = data_pre['distance'] + (data_pre[dimension] - temp_center[dimension]) ** 2
            data_pre['distance'] = numpy.sqrt(data_pre['distance'])
            radius = data_pre['distance'].mean()
            radius = radius_0 if radius >= radius_0 else radius
            # Construct the subset
            subset_index = data_pre.loc[data_pre['distance'] <= radius].index
            subset = data_pre.loc[subset_index].copy()
            data_pre = pandas.concat([data_pre, center], axis=0)
            subset = pandas.concat([subset, center], axis=0)

            # Optimization the subset
            # Optimization zoning anchor
            gravity = subset.mean()
            subset['distance'] = 0
            for dimension in self.dimension:
                subset['distance'] = subset['distance'] + (subset[dimension] - gravity[dimension]) ** 2
            subset['distance'] = numpy.sqrt(subset['distance'])
            temp_center = subset.loc[subset['distance'].idxmin()].copy()
            center = temp_center.to_frame()
            center = pandas.DataFrame(center.values.T, columns=center.index, index=center.columns)
            # Optimization zoning radius
            subset = subset.drop(index=center.index)
            subset['distance'] = 0
            for dimension in self.dimension:
                subset['distance'] = subset['distance'] + (subset[dimension] - temp_center[dimension]) ** 2
            subset['distance'] = numpy.sqrt(subset['distance'])
            radius = subset['distance'].mean()
            radius = 0 if radius is numpy.NaN else radius
            # Optimization the subset
            data_pre['distance'] = 0
            for dimension in self.dimension:
                data_pre['distance'] = data_pre['distance'] + (data_pre[dimension] - temp_center[dimension]) ** 2
            data_pre['distance'] = numpy.sqrt(data_pre['distance'])
            subset_index = data_pre.loc[data_pre['distance'] <= radius].index
            subset = data_pre.loc[subset_index].copy()
            # Updating data
            self.csp_center = pandas.concat([self.csp_center, center], axis=0)
            self.csp_center.loc[center.index, 'radius'] = radius
            self.csp_center.loc[center.index, 'amount'] = int(subset.shape[0])
            self.csp_center.loc[center.index, 'subset_id'] = subset_id
            subset['subset_id'] = subset_id
            data_pre = data_pre.drop(index=subset_index, inplace=False)
            self.csp_data = pandas.concat([self.csp_data, subset], axis=0)

        print('After Constructive Sample Partition：')
        print('csp_data:', self.csp_data.shape, '\n', self.csp_data)
        print('-------------------------------------------')
        print('center:', self.csp_center.shape, '\n', self.csp_center)
        print('-------------------------------------------')

    # Cleaning hybrid subsets
    def clean(self):
        self.csp_center['majority_amount'] = 0
        self.csp_center['minority_amount'] = 0
        for center_index, center in self.csp_center.iterrows():
            temp_data = self.csp_data.loc[self.csp_data['subset_id'] == center['subset_id']].copy()
            self.csp_center.loc[center_index, 'majority_amount'] = int(temp_data.loc[temp_data['label'] == self.maj_label].shape[0])
            self.csp_center.loc[center_index, 'minority_amount'] = int(temp_data.loc[temp_data['label'] == self.min_label].shape[0])
            temp_majority = temp_data.loc[temp_data['label'] == self.maj_label].copy()
            temp_minority = temp_data.loc[temp_data['label'] == self.min_label].copy()
            # Heterogeneous subset cleaning
            if temp_majority.shape[0] > 0 and temp_minority.shape[0] > 0:
                temp_majority['distance'] = 0
                for majority_index, majority in temp_majority.iterrows():
                    temp_minority2 = temp_minority.copy()
                    temp_minority2['distance'] = 0
                    for dimension in self.dimension:
                        temp_minority2['distance'] = temp_minority2['distance'] + (temp_minority2[dimension] - majority[dimension]) ** 2
                    temp_minority2['distance'] = numpy.sqrt(temp_minority2['distance'])
                    temp_majority.loc[majority_index, 'distance'] = temp_minority2['distance'].sum()
                distance_avg = temp_majority['distance'].mean()
                remove_maj_index = temp_majority.loc[temp_majority['distance'] < distance_avg].index
                self.csp_data.drop(index=remove_maj_index, inplace=True)
                self.csp_center.loc[center_index, 'amount'] = self.csp_center.loc[center_index, 'amount'] - remove_maj_index.shape[0]
                self.csp_center.loc[center_index, 'majority_amount'] = self.csp_center.loc[center_index, 'majority_amount'] - remove_maj_index.shape[0]
                temp_center = self.csp_center.loc[center_index].copy()
                center = temp_center.to_frame()
                center = pandas.DataFrame(center.values.T, columns=center.index, index=center.columns)
                self.hybrid_subset = pandas.concat([self.hybrid_subset, center], axis=0)
        print('After cleaning hybrid subsets：')
        print('csp_data:', self.csp_data.shape, '\n', self.csp_data)
        print('-------------------------------------------')
        print('hybrid_subset:', self.hybrid_subset.shape, '\n', self.hybrid_subset)
        print('-------------------------------------------')

    # Generating new samples
    def generate(self):
        # Calculate imbalanced ratio IR
        temp_majority = self.csp_data.loc[self.csp_data['label'] == 1].copy()
        temp_minority = self.csp_data.loc[self.csp_data['label'] == 0].copy()
        self.imbalance_rate = temp_majority.shape[0] / temp_minority.shape[0]

        # Calculate the number of synthetic samples
        if self.imbalance_rate <= 1 or self.hybrid_subset.shape[0] == 0:
            return
        else:
            radius_sum = self.hybrid_subset['radius'].sum()
            minority_sum = self.hybrid_subset['minority_amount'].sum()
            self.hybrid_subset['power'] = (self.hybrid_subset['radius'] / radius_sum) * (self.hybrid_subset['minority_amount'] / minority_sum)
            power_sum = self.hybrid_subset['power'].sum()
            self.hybrid_subset['power'] = self.hybrid_subset['power'] / power_sum
            self.hybrid_subset['syn_number'] = numpy.around(self.hybrid_subset['power'] * (temp_majority.shape[0] - temp_minority.shape[0]))
            self.hybrid_subset['syn_number'] = self.hybrid_subset['syn_number'].astype(int)

        # Calculate synthetic seed
        for row_index, row in self.hybrid_subset.iterrows():
            hybrid_data = self.csp_data.loc[self.csp_data['subset_id'] == row['subset_id']].copy()
            temp_minority = hybrid_data.loc[hybrid_data['label'] == 0].copy()
            temp_minority['syn_number'] = numpy.around(row['syn_number'] / row['minority_amount'])
            temp_minority['nn_number'] = 5
            temp_minority['nn_number'] = temp_minority['nn_number'].astype(int)
            self.seed = pandas.concat([self.seed, temp_minority], axis=0)

        # Find auxiliary samples
        for row_index, row in self.seed.iterrows():
            data_pre = self.csp_data.copy()
            data_pre.drop(index=row_index, inplace=True)
            data_pre['distance'] = 0
            for dimension in self.dimension:
                data_pre['distance'] = data_pre['distance'] + (data_pre[dimension] - row[dimension]) ** 2
            data_pre['distance'] = numpy.sqrt(data_pre['distance'])
            data_pre = data_pre.sort_values(by='distance', ascending=True)
            temp_auxiliary = data_pre.iloc[0:int(row['nn_number'])].copy()
            self.auxiliary[row_index] = temp_auxiliary

        # Generate
        for row_index, row in self.seed.iterrows():
            self.synthesize_smote(row_index)

        print('After generating new samples：')
        print('seed:', self.seed.shape, '\n', self.seed)
        print('-------------------------------------------')
        print('synthesis:', self.synthesis.shape, '\n', self.synthesis)
        print('-------------------------------------------')

    # synthesize method
    def synthesize_smote(self, seed_index):
        seed = self.seed.loc[seed_index]
        for _ in range(int(seed['syn_number'])):
            weights = numpy.random.rand(len(self.dimension))
            weights = numpy.concatenate((weights, [0]), axis=0)
            synthesis = pandas.Series(weights)
            synthesis = synthesis.to_frame()
            synthesis = pandas.DataFrame(synthesis.values.T, columns=self.data.columns)

            auxiliary_set = self.auxiliary[seed_index]
            temp_auxiliary = auxiliary_set.sample(n=1, replace=False)
            temp_auxiliary = temp_auxiliary.squeeze()
            for dimension in self.dimension:
                synthesis[dimension] = seed[dimension] + synthesis[dimension] * (temp_auxiliary[dimension] - seed[dimension])
            self.synthesis = pandas.concat([self.synthesis, synthesis], axis=0)

    # output
    def output(self):
        del self.csp_data['distance']
        del self.csp_data['subset_id']
        self.result = pandas.concat([self.csp_data, self.synthesis], axis=0)
        self.result['label'] = self.result['label'].astype('int')

        print('Output：')
        print('result:', self.result.shape, '\n', self.result)
        print('-------------------------------------------')

        self.result = self.result.to_numpy().copy()
        x_resampled = self.result[:, 0:-1:1].copy()
        y_resampled = self.result[:, -1].copy()
        return x_resampled, y_resampled

    # Resampling
    def fit_resample(self, X, y):
        data_set = numpy.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
        self.data = pandas.DataFrame(data=data_set)
        self.data.rename(columns={self.data.shape[1] - 1: 'label'}, inplace=True)
        count = collections.Counter(y)
        label_1, label_2 = set(count.keys())
        self.min_label, self.maj_label = (label_1, label_2) if count[label_1] < count[label_2] else (label_2, label_1)
        self.min_data = self.data.loc[self.data['label'] == self.min_label].copy()
        self.maj_data = self.data.loc[self.data['label'] == self.maj_label].copy()
        self.dimension = self.data.columns[0:-1]
        self.imbalance_rate = self.maj_data.shape[0] / self.min_data.shape[0]
        self.result = pandas.DataFrame(columns=self.data.columns)

        self.constructiveSamplePartition()
        self.clean()
        self.generate()
        x_resampled, y_resampled = self.output()
        return x_resampled, y_resampled


if __name__ == '__main__':
    data = pandas.read_csv('xxx.csv', header=None)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
    data = scaler.fit_transform(data)
    x_train = data[:, :-1].copy()
    y_train = data[:, -1].copy()
    method = CSPHS()
    x_resampled, y_resampled = method.fit_resample(x_train, y_train)
