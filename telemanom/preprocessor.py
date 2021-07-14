import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger('telemanom')

class Preprocessor:
    def __init__(self, config, chan_id, train, test):
        """
        Preprocess data in preparation for modeling.

        Args:
            config (obj): Config object containing parameters for processing
            chan_id (str): channel id
            train (arr): numpy array containing raw train data
            test (arr): numpy array containing raw test data

        Attributes:
            id (str): channel id
            config (obj): see Args
            train (arr): train data loaded from .npy file
            test(arr): test data loaded from .npy file
        """

        self.id = chan_id
        self.config = config
        self.train = None
        self.test = None
        self.scaler = None

        if self.config.scale:
            self.scale()

    def scale(self):
        """Min/Max or Standard Scale
        Remove outliers, etc.
        """
        if self.config.scaler == 'min_max':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        self.train = self.scaler.fit_transform(self.train)
        self.test = self.scaler.transform(self.test)