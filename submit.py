import pandas as pd
from pandas.core.frame import DataFrame

data_frame = pd.read_csv('final_result.csv').values
flabel = []
Id = []
count = 0
for label in data_frame:
    tlabel = 0
    if label[0] - label[1] > 0.95:
        tlabel = 1
    else:
        tlabel = 0
    flabel.append(tlabel)
    Id.append(count)
    count += 1
data_frame = DataFrame({'Id':Id, 'Expected':flabel})
data_frame.to_csv('./submission.csv', index = False)

