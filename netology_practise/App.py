import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.font as tkfont
from sklearn.preprocessing import LabelEncoder

canvas = None

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        model = load_model()  
        process_and_plot_data(model, file_path, root)

def load_model():
    #Check your path to model
    model = joblib.load('d:\\model.pkl')
    return model

def process_and_plot_data(model, file_path, root):
    try:
        df_test = pd.read_csv(file_path)

        if 'Weekly_Sales' in df_test.columns.values.tolist():
            df_test = df_test.drop(columns=['Weekly_Sales'])
            name_of_graph = 'Прошлые продажи, $'
        else:
            name_of_graph = 'Прогноз будущих продаж, $'
        df_test["IsHoliday"] = df_test["IsHoliday"].astype(int)

        df_test['Date'] = pd.to_datetime(df_test['Date'])
        df_test = df_test.sort_values(by='Date', ascending=True)

        df_date = df_test.copy()
        df_test['day'] = df_test['Date'].dt.day
        df_test["day"] = df_test["day"].astype(int)

        df_test['month'] = df_test['Date'].dt.month
        df_test["month"] = df_test["month"].astype(int)

        df_test['year'] = df_test['Date'].dt.year
        df_test["year"] = df_test["year"].astype(int)

        df_test['week'] = df_test['Date'].dt.isocalendar().week
        df_test["week"] = df_test["week"].astype(int)

        df_test = df_test.drop(columns=['Date'])

        df_test['Discount'] = df_test['MarkDown1'] + df_test['MarkDown2'] + df_test['MarkDown3'] + df_test['MarkDown4'] + df_test['MarkDown5']
        df_test = df_test.drop(columns=['MarkDown1','MarkDown2','MarkDown3', 'MarkDown4', 'MarkDown5'])

        median_value = df_test['CPI'].median()
        df_test.fillna({'CPI': median_value}, inplace=True)

        median_value = df_test['Unemployment'].median()
        df_test.fillna({'Unemployment': median_value}, inplace=True)

        df_test.fillna(0, inplace=True)

        le = LabelEncoder()
        df_test['Type'] = le.fit_transform(df_test['Type'])

        df_test = df_test.drop(columns=['Type', 'Unemployment', 'CPI'])
        predictions = model.predict(df_test)

        df_predictions = pd.DataFrame({'Date': df_date['Date'], 'Sales': predictions})
        df_grouped = df_predictions.groupby('Date', as_index=False).sum()

        display_graph(df_grouped, root, name_of_graph)
        
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось обработать данные: {e}")

def display_graph(df_grouped, root, name_of_graph):
    global canvas
    try:
        if canvas is None:
            fig, ax = plt.subplots()
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        else:
            ax = canvas.figure.axes[0]

        ax.plot(df_grouped['Date'], df_grouped['Sales'], label=name_of_graph)
        ax.set_xlabel('Дата')
        ax.set_ylabel('Продажи, $')

        ax.legend()
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось отобразить график: {e}")

root = tk.Tk()
root.title("Прогноз продаж")
root.geometry("1200x800")

header_font = tkfont.Font(family="Helvetica", size=18, weight="bold")
description_font = tkfont.Font(family="Helvetica", size=12)
button_font = tkfont.Font(family="Helvetica", size=10)

header = tk.Label(root, text="Приложение для прогноза продаж", font=header_font)
header.pack(pady=10)

description = tk.Label(root, text="Прогнозирование продаж является важной задачей для бизнеса, \nтак как позволяет более точно планировать производство, \nзапасы, маркетинговые кампании и финансовые ресурсы.\n Загрузите CSV файл с данными продаж для прогнозирования.", font=description_font)
description.pack(pady=10)

upload_btn = tk.Button(root, text="Загрузить CSV файл", font=button_font, command=upload_file, bg="#007BFF", fg="white")
upload_btn.pack(pady=20)

author_label = tk.Label(root, text="Авторы: Захаров Иван, Лукьянчик Ян, Савинов Сергей, Шорохова Юлия", font=description_font, fg="gray")
author_label.pack(side="bottom", pady=10)

root.mainloop()
