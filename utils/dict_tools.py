
import matplotlib.pyplot as plt


class DictTools(object):

    @staticmethod
    def update_dict_with_lists(target_dict, new_values_dict):
        for key in new_values_dict.keys():
            if key not in target_dict.keys():
                target_dict[key] = list()
            target_dict[key].append(new_values_dict[key])

    @staticmethod
    def log_dict_avg_sorted(logger, values_dict):

        logger.log('-- results')

        total_keys = len(values_dict.keys())
        plt.subplots(num=None, figsize=(15, 6), dpi=120, facecolor='w', edgecolor='k')

        plotted = False
        for key_idx, key in enumerate(sorted(values_dict.keys(), reverse=True)):
            if type(values_dict[key]) == list:
                plotted = True
                list_sum = sum(values_dict[key])
                list_len = len(values_dict[key])
                logger.log('{key} : {avg_value}'.format(key=key, avg_value=list_sum / list_len))

                plt.subplot(100 + 10 * total_keys + key_idx + 1)
                plt.hist(values_dict[key], bins=100, log=True)
                plt.title(key)

            else:
                logger.log('{key} : {key_value}'.format(key=key, key_value=values_dict[key]))

        if plotted:
            plt.show()

