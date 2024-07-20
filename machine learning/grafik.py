import pandas as pd
import matplotlib.pyplot as plt

# CSV dosyalarını oku
loss_data = pd.read_csv('loss_data.csv')

# Verileri grupla ve uygun şekilde ayır
methods = loss_data['method'].unique()

# Süre vs. Loss grafiği
plt.figure(figsize=(14, 6))

for method in methods:
    method_data = loss_data[loss_data['method'] == method]
    plt.plot(method_data['time'], method_data['loss'], label=f'{method}')

plt.xlabel('Time (seconds)')
plt.ylabel('Loss')
plt.title('Time vs. Loss')
plt.legend()
plt.grid(True)
plt.show()

# Epoch vs. Loss grafiği
plt.figure(figsize=(14, 6))

for method in methods:
    method_data = loss_data[loss_data['method'] == method]
    plt.plot(method_data['epoch'], method_data['loss'], label=f'{method}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs. Loss')
plt.legend()
plt.grid(True)
plt.show()
