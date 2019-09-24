import joblib
import matplotlib.pyplot import plt
from matrix_analyse_report_anomaly import *

model = joblib.load(model.pkl)

yhat = model.predict(x_test)



plt.plot(x_list, rmses)
plt.ylabel("Errors Values")
file_number = re.findall('\d+', file)
print("the file_number is:", file_number)
plt.title(file_number[0] + ' ' + 'Errors Distribution')
# plt.title(file + ' ' + 'Errors Distribution')
plt.show()