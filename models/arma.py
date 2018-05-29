

from statsmodels.tsa.arima_model import ARMA


class ARMAModel(object):

    SOLVERS = [
        'bfgs', 'powell', 'nm', 'lbfgs', 'newton', 'cg', 'ncg'
    ]

    METHODS = [
        'css-mle', 'mle', 'css'
    ]

    def __init__(self, logger, train_set, p, q, with_c):
        self.__logger = logger

        # set AR component order
        self.__p = p

        # set MA component order
        self.__q = q

        # set trend setting
        self.__trend = 'c' if with_c else 'nc'

        # init empty members
        self.__train_set = train_set
        self.__model = None

        # learn model params
        self.__learn_model_params()

    def __learn_model_params(self, print_settings=False, start_params=None):
        # validate data
        if len(self.__train_set) > 1 and len(set(self.__train_set)) == 1:
            raise Exception("Can't fit model since all history values are the same")

        # create model instance
        model = ARMA(self.__train_set, order=(self.__p, self.__q))

        # fit model - try every possible setting combination
        model_fit = None
        for trans_params_mode in [True, False]:
            for solver in ARMAModel.SOLVERS:
                for method in ARMAModel.METHODS:
                    try:
                        model_fit = \
                            model.fit(
                                disp=0,
                                trend=self.__trend,
                                method=method,
                                solver=solver,
                                transparams=trans_params_mode,
                                start_params=start_params
                            )
                        if print_settings:
                            self.__logger.log(
                                'settings: method={method}, solver={solver}, transparams={transparams}'
                                .format(method=method, solver=solver, transparams=trans_params_mode),
                                should_print=False
                            )
                        break

                    except Exception as ex:
                        continue

                if model_fit is not None:
                    break

            if model_fit is not None:
                break

        if model_fit is None:
            raise Exception('No settings worked')

        self.__model = model_fit
        self.__logger.log(self.__model.params)

    def get_fitted_values(self):
        return self.__model.fittedvalues
