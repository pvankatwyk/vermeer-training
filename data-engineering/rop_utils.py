import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')


class ROPData:
    def __init__(self):
        self.filepath = ''
        self.data = pd.DataFrame()

    def upload(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        return self

    def process(self):
        data = self.data
        data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
        n = len(data)
        deltaTime = np.zeros(n) * np.nan
        forward = [True] * n

        # Calculate time stamp differences
        for i in range(1, n):
            deltaTime[i] = (data['TimeStamp'][i] - data['TimeStamp'][i - 1]).total_seconds()
            forward[i] = True if data['RodCount'][i] > data['RodCount'][i - 1] else False

        # average of 2 and 3 for 1st time point only
        deltaTime[0] = np.mean(deltaTime[1:3])
        data['ROP (ft/min)'] = (60 * 10.0 / deltaTime)
        data['deltaTime'] = deltaTime
        data = data[forward]
        # Drops rows with no time change
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["ROP (ft/min)"], how="all")
        data = data.reset_index(drop=True)
        self.data = data
        return self

    def filter(self, rop_greater_than=None, rop_less_than=None):
        data = self.data.loc[
            (self.data['Rotation Speed Max (rpm)'] > 0) & (self.data['Rotation Torque Max (ft-lb)'] > 0)]
        if rop_greater_than is not None:
            data = data.loc[data['ROP (ft/min)'] > rop_greater_than]
        if rop_less_than is not None:
            data = data.loc[data['ROP (ft/min)'] < rop_less_than]
        self.data = data
        return self

    def add_quartiles(self, num_divisions=4):
        step = 1 / num_divisions
        q = list(np.arange(step, 1, step))
        quartile_cutoffs = np.quantile(self.data['ROP (ft/min)'], q=q)

        def separate_quartiles(x, quartile_vec):
            # make a list of range tuples
            l = []
            for i, val in enumerate(quartile_vec):
                l.append((quartile_vec[i - 1], val))

            # fix the first and last tuples
            l[0] = (0, l[0][1])
            l.append((l[len(l) - 1][1], 15))

            # make a dictionary with the quartiles and their tuples
            d = dict((i + 1, l[i]) for i in range(len(l)))

            # look up the quartile
            for i in d:
                rng = d.get(i)
                if rng[0] <= x < rng[1]:
                    quartile = i

            return quartile

        self.data['quartiles'] = self.data['ROP (ft/min)'].apply(lambda x: separate_quartiles(x, quartile_cutoffs))
        # self.data['quartiles'] = pd.qcut(self.data['ROP (ft/min)'], num_divisions,
        #                                  labels=np.arange(start=1, stop=num_divisions + 1)[::-1],
        #                                  duplicates='drop')
        return self

    def list_columns(self):
        return self.data.columns

#     def train_test_split(self, train_proportion):
        
    def train_model(self, model, train_proportion=0.8, features='all'):
        if features.lower() == 'all':
            data = self.data.select_dtypes([float, int]).drop(
                columns=['Latitude', 'Longitude', 'RodCount', 'deltaTime'])
        else:
            assert 'ROP (ft/min)' in features, "Features list must contain ROP measurement."
            data = self.data[features].select_dtypes([float, int]).drop(
                columns=['Latitude', 'Longitude', 'RodCount', 'deltaTime'])

        X = data.drop(columns=['ROP (ft/min)'])
        y = data['ROP (ft/min)']

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, random_state=42)

        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(test_predictions, y_test))

        return model, test_predictions, rmse

    def pdp_plots(self, trained_model, save_dir=None):
        data = self.data.select_dtypes([float, int]).drop(columns=['Latitude', 'Longitude', 'RodCount', 'deltaTime'])
        X = data.drop(columns=['ROP (ft/min)'])
        y = data['ROP (ft/min)']

        # Median Dataset Plots
        prediction_list = []
        delta_column_list = []
        for column in X.columns:
            # Create array for the variable being changed
            delta_column = np.linspace(int(min(X[column])), int(max(X[column])), 100000)
            df = pd.DataFrame(delta_column)
            # Create columns for all variables with the median values
            df['Rotation Speed Max (rpm)'] = np.median(X['Rotation Speed Max (rpm)'])
            df['Rotation Torque Max (ft-lb)'] = np.median(X['Rotation Torque Max (ft-lb)'])
            df['Thrust Force Max (lbf)'] = np.median(X['Thrust Force Max (lbf)'])
            df['Mud Flow Rate Avg (gpm)'] = np.median(X['Mud Flow Rate Avg (gpm)'])
            df['Mud Pressure Max (psi)'] = np.median(X['Mud Pressure Max (psi)'])
            df['Thrust Speed Avg (ft/min)'] = np.median(X['Thrust Speed Avg (ft/min)'])
            df['Pull Force Maximum (lbf)'] = np.median(X['Pull Force Maximum (lbf)'])
            df['Pull Speed Average (ft/min)'] = np.median(X['Pull Speed Average (ft/min)'])
            df['Drill String Length (ft)'] = np.median(X['Drill String Length (ft)'])
            # Delete the median column for the column being changed
            del df[column]
            # Rename the delta_column array to the column of interest
            df = df.rename(columns={0: column})
            # Reorder the columns to the match model requirements
            df = df[
                ['Rotation Speed Max (rpm)', 'Rotation Torque Max (ft-lb)', 'Thrust Force Max (lbf)',
                 'Mud Flow Rate Avg (gpm)',
                 'Mud Pressure Max (psi)', 'Thrust Speed Avg (ft/min)', 'Pull Force Maximum (lbf)',
                 'Pull Speed Average (ft/min)',
                 'Drill String Length (ft)']]

            # Predict with model and plot
            prediction = trained_model.predict(df)

            prediction_list.append([column, prediction])
            delta_column_list.append([column, delta_column])

        # Loops through all
        prediction = dict(prediction_list)
        delta_column_dict = dict(delta_column_list)
        for key in prediction:
            quartiles = np.quantile(prediction[key], q=[0.25, 0.5, 0.75])
            quartile1 = quartiles[2]
            quartile2 = quartiles[1]
            quartile3 = quartiles[0]

            q1_y = prediction[key][prediction[key] >= quartile1]
            q2_y = prediction[key][(prediction[key] < quartile1) & (prediction[key] >= quartile2)]
            q3_y = prediction[key][(prediction[key] < quartile2) & (prediction[key] >= quartile3)]
            q4_y = prediction[key][(prediction[key] < quartile3) & (prediction[key] >= min(prediction[key]))]

            x1 = delta_column_dict[key][(prediction[key] >= quartile1).nonzero()[0]]
            x2 = delta_column_dict[key][((prediction[key] < quartile1) & (prediction[key] >= quartile2)).nonzero()[0]]
            x3 = delta_column_dict[key][((prediction[key] < quartile2) & (prediction[key] >= quartile3)).nonzero()[0]]
            x4 = delta_column_dict[key][
                ((prediction[key] < quartile3) & (prediction[key] >= min(prediction[key]))).nonzero()[0]]

            plt.figure(figsize=(10, 5))
            plt.plot(x1, q1_y, '.', markersize=8, label='Q1')
            plt.plot(x2, q2_y, '.', markersize=8, label='Q2')
            plt.plot(x3, q3_y, '.', markersize=8, label='Q3')
            plt.plot(x4, q4_y, '.', markersize=8, label='Q4')
            plt.plot(delta_column_dict[key], prediction[key], 'k-', linewidth=0.5)
            plt.xlabel(key)
            plt.ylabel('ROP (ft/min)')
            plt.title(key)
            plt.legend()

            if save_dir is not None:
                fp = save_dir + str(key) + '.jpg'
                if not os.path.exists(fp):
                    plt.savefig(fp)

            plt.show()
