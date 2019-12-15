import numpy as np

class FCA:

    def __init__(self, debug=0):
        self.debug = debug
        self.column_value = []

    #####################################################################

    def _concat_elements(self, element1, element2):

        """ element1 and element2 must by list | return list of lists """

        r_concat = list()

        for i in range(len(element1)):
            if element1[i] == 1:
                if element1[i] == element2[i]:
                    result = 1
                    r_concat.append(result)
                else:
                    result = 0
                    r_concat.append(result)
            elif element1[i] == 2:
                if element1[i] == element2[i]:
                    result = 0
                    r_concat.append(result)
                else:
                    result = 0
                    r_concat.append(result)
        return r_concat

    def training(self, validation_data, train_data):

        """ element1 and element2 must by list | return list of lists """

        train_list = list()

        if self.debug == 1:
            print("The intesection of test and train data is :")

        for i in range(len(validation_data)):
            for j in range(len(train_data)):
                result = self._concat_elements(validation_data[i], train_data[j])
                if self.debug == 1:
                    print(validation_data[i], train_data[j], result)
                train_list.append(result)

        return train_list

    #####################################################################

    def _comparing_rows(self, obj1, obj2):

        """obj1 , obj2 list | return list"""
        _comparing_list = list()

        for i in range(len(obj1)):
            if obj1[i] == 0:
                _comparing_list.append('*')
            elif obj1[i] == obj2[i]:
                _comparing_list.append(1)
            elif obj1[i] != obj2[i]:
                _comparing_list.append(0)

        return _comparing_list

    def comparing(self, result_of_training, train_data):

        comparing_list = list()

        if self.debug == 2:
            waiting(len(result_of_training), 100)

        # result_of_training
        for row in result_of_training:
            # Train data
            for train in train_data:
                result = self._comparing_rows(row, train)
                comparing_list.append(result)
                self.column_value.append(result)

        cutting = len(comparing_list) / len(train_data)

        return np.array_split(comparing_list, cutting)

    #####################################################################

    def weight_of_result(self, result_of_comparing):

        weight_of_result = list()

        for i, arr in enumerate(result_of_comparing):
            buffer = list()
            for j, row in enumerate(arr):
                if '0' not in row:
                    buffer.append(1)
                else:
                    buffer.append(0)

            _sum = 100 * (np.sum(np.array(buffer)) / len(buffer))
            weight_of_result.append([buffer, _sum])

        return weight_of_result

    def compar_positive_and_negative(self, positiv, negativ, pos_weight=1, neg_weight=1):

        positive = 0
        negative = 0
        for i in range(len(positiv)):
            if float(positiv[i][1] * pos_weight) >= float(negativ[i][1] * neg_weight):
                positive += 1
            elif float(positiv[i][1] * pos_weight) < float(negativ[i][1] * neg_weight):
                negative += 1

        return positive, negative

    #####################################################################

    def result(self, count_positive, count_negative):

        result_count = [count_positive[0] + count_negative[0], count_positive[1] + count_negative[1]]
        result_positive = 100 * (result_count[0] / np.sum(result_count))
        result_negative = 100 * (result_count[1] / np.sum(result_count))
        return result_positive, result_negative

    #####################################################################

    def fca(self, test, positive, negative, pos_weight=1, neg_weight=1):

        if self.debug == 1:
            print("Postive")

        # comparing positive with positive
        result_positive = self.training(test, positive)
        result_pos_pos = self.comparing(result_positive, positive)
        weight_pos_pos = self.weight_of_result(result_pos_pos)

        # comparing positive with negative
        result_pos_neg = self.comparing(result_positive, negative)
        weight_pos_neg = self.weight_of_result(result_pos_neg)

        # count
        count_positive = self.compar_positive_and_negative(weight_pos_pos, weight_pos_neg, pos_weight, neg_weight)

        if self.debug:
            print("\n The result of intersection result of positive element and positive train is :\n\n",
                  np.array(result_pos_pos))
            print("\n The weigth is :\n\n", np.array(weight_pos_pos))
            print("\n The result of intersection result of positive element and negative train is :\n\n",
                  np.array(result_pos_neg))
            print("\n The weigth is :\n\n", np.array(weight_pos_neg))
            print("\n The count is : \n\n", np.array(count_positive))

        ####################################

        if self.debug == 1:
            print("\n")
            print("###################")
            print("\n")
            print("Negative\n")

        # comparing negative with positive
        result_negative = self.training(test, negative)
        result_neg_pos = self.comparing(result_negative, positive)
        weight_neg_pos = self.weight_of_result(result_neg_pos)

        # comparing positive with negative
        result_neg_neg = self.comparing(result_negative, negative)
        weight_neg_neg = self.weight_of_result(result_neg_neg)

        # count
        count_negative = self.compar_positive_and_negative(weight_neg_pos, weight_neg_neg, pos_weight, neg_weight)

        if self.debug:
            print("\n The result of intersection result of negative element and positive train is :\n\n",
                  np.array(result_neg_pos))
            print("\n The weigth is :\n\n", np.array(weight_neg_pos))
            print("\n The result of intersection result of negative element and negative train is :\n\n",
                  np.array(result_neg_neg))
            print("\n The weigth is :\n\n", np.array(weight_neg_neg))
            print("\n The count is : \n\n", np.array(count_negative))

        ####################################

        result_count = self.result(count_positive, count_negative)

        return result_count

