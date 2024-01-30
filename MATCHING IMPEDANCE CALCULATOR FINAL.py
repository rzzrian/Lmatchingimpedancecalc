import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import skrf as rf
import numpy as np
from scipy.optimize import minimize

def init_blank_screen(canvas):
    fig, ax = plt.subplots(figsize=(12, 8))  #adjust size
    canvas.get_tk_widget().destroy()  # Remove any previous canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(column=2, row=0, rowspan=10, padx=15, pady=15)
    canvas.draw()

def calculate_and_plot(ZL_real, ZL_imag, Zo, frequency, canvas):
    init_blank_screen(canvas)

    ZL = complex(ZL_real, ZL_imag)
    Z0 = Zo
    x = 1
    y = 0
    ZL_normalized = ZL / Z0
    s1 = ((ZL_normalized - 1) / (ZL_normalized + 1))
    freq = rf.Frequency(frequency * 1e6, unit='Hz')  # Convert to Hz
    obj = rf.Network(frequency=freq, s=np.ones(3) * s1, z0=Z0)

    admitansi = 1 / ZL_normalized
    
    # Matching network optimization
    center_frequency = frequency
    freq = f'{center_frequency}MHz'
    frequency_bandwidth = 100  # in MHz, adjust as needed
    frequency_start = center_frequency - frequency_bandwidth/2
    frequency_stop = center_frequency + frequency_bandwidth/2
    frequency = rf.Frequency(start=frequency_start, stop=frequency_stop, npoints=401, unit='MHz')

    # Define the transmission line Media
    line = rf.DefinedGammaZ0(frequency=frequency, z0=Zo)
    ZL = ZL_real + 1j * ZL_imag

    # Define the load Network
    load = line.load(rf.zl_2_Gamma0(Zo, ZL))

    # Matching network functions
    def matching_network_LC_1(L, C):
        ' L and C in nH and pF'
        return line.inductor(L*1e-9)**line.shunt_capacitor(C*1e-12)**load

    def matching_network_LC_2(L, C):
        ' L and C in nH and pF'
        return line.capacitor(C*1e-12)**line.shunt_inductor(L*1e-9)**load

    # Objective functions for optimization
    def optim_fun_1(x, f0=freq):
        _ntw = matching_network_LC_1(*x)
        return np.abs(_ntw[freq].s).ravel()

    def optim_fun_2(x, f0=freq):
        _ntw = matching_network_LC_2(*x)
        return np.abs(_ntw[freq].s).ravel()

    # Initial guess values and bounds
    L0 = 10  # nH
    C0 = 1   # pF
    x0 = (L0, C0)
    L_minmax = (1, 100)  # nH
    C_minmax = (0.1, 10)  # pF

    # Optimization using scipy
    res1 = minimize(optim_fun_1, x0, bounds=(L_minmax, C_minmax))
    res2 = minimize(optim_fun_2, x0, bounds=(L_minmax, C_minmax))

    # Display optimization results
    label_hasil_induktor = ttk.Label(input_frame, text=f"L={res1.x[0]:.2f} nH", font=2)
    label_hasil_induktor.grid(column=1, row=19, padx=1, pady=1, sticky='W')

    label_hasil_capasitor = ttk.Label(input_frame, text=f"C={res1.x[1]:.2f} pF", font=2)
    label_hasil_capasitor.grid(column=1, row=20, padx=1, pady=1, sticky='W')

    label_hasil_induktor = ttk.Label(input_frame, text=f"L={res2.x[0]:.2f} nH", font=2)
    label_hasil_induktor.grid(column=1, row=32, padx=1, pady=1, sticky='W')
    
    label_hasil_capasitor = ttk.Label(input_frame, text=f"C={res2.x[1]:.2f} pF", font=2)
    label_hasil_capasitor.grid(column=1, row=33, padx=1, pady=1, sticky='W')

    # Use the FigureCanvasTkAgg to embed the matplotlib figure in Tkinter
    fig, ax = plt.subplots(figsize=(12, 8))  #adjust size
    scatter = obj.plot_s_smith(ax=ax, draw_labels=True, marker='o', show_legend=False)
    plt.scatter(0.0, 0.0, marker='o', color = 'green')
    plt.scatter(admitansi.real, admitansi.imag, marker='o', color='red')
    
    # Update the canvas with the new figure
    canvas.get_tk_widget().destroy()  # Remove any previous canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(column=2, row=0, rowspan=10, padx=15, pady=15)

    # Menambahkan label di pojok kanan atas
    ax.text(0.90, 0.90, f'Z.Normalisasi: {ZL_normalized.real:.2f} + j {ZL_normalized.imag:.2f}', transform=ax.transAxes, color='blue', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))
    ax.text(0.90, 0.85, f'Admitansi: {admitansi.real:.2f} + j {admitansi.imag:.2f}', transform=ax.transAxes, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
    # Menambahkan judul
    ax.set_title('Smith Chart', fontsize=16)
    canvas.draw()


# Membuat GUI
root = tk.Tk()
root.title("Matching Impedansi Kalkulator")

# Membuat frame untuk entry dan label
input_frame = ttk.Frame(root)
input_frame.grid(column=0, row=0, padx=10, pady=10, sticky='W')

# Label dan Entry untuk ZL_real
label_ZL_real = ttk.Label(input_frame, text="Matching Impedance Calculator", foreground='red', font=6)
label_ZL_real.grid(column=0, row=0, padx=5, pady=5, sticky='W')

# Label dan Entry untuk ZL_real
label_ZL_real = ttk.Label(input_frame, text="Impedansi Beban (ZL_real) Ω :", font=1)
label_ZL_real.grid(column=0, row=1, padx=5, pady=5, sticky='W')
entry_ZL_real = ttk.Entry(input_frame)
entry_ZL_real.grid(column=1, row=1, padx=5, pady=5, sticky='W')

# Label dan Entry untuk ZL_imag
label_ZL_imag = ttk.Label(input_frame, text="Impedansi Beban (ZL_Imag) Ω :", font=1)
label_ZL_imag.grid(column=0, row=2, padx=5, pady=5, sticky='W')
entry_ZL_imag = ttk.Entry(input_frame)
entry_ZL_imag.grid(column=1, row=2, padx=5, pady=5, sticky='W')

# Label dan Entry untuk Zo 
label_Zo = ttk.Label(input_frame, text="Impedansi karakteristik (Zo) Ω :", font=1)
label_Zo.grid(column=0, row=3, padx=5, pady=5, sticky='W')
entry_Zo = ttk.Entry(input_frame)
entry_Zo.grid(column=1, row=3, padx=5, pady=5, sticky='W')

# Label dan Entry untuk frequency
label_frequency = ttk.Label(input_frame, text="Frequency (MHz) :", font=1)
label_frequency.grid(column=0, row=4, padx=5, pady=5, sticky='W')
entry_frequency = ttk.Entry(input_frame)
entry_frequency.grid(column=1, row=4, padx=5, pady=5, sticky='W')

# Tombol Submit
submit_button = ttk.Button(input_frame, text="Submit", 
                           command=lambda: calculate_and_plot
                           (float(entry_ZL_real.get()), float(entry_ZL_imag.get()),
                            float(entry_Zo.get()), float(entry_frequency.get()),
                            canvas))
submit_button.grid(column=0, row=5, columnspan=2, pady=10, sticky='WE')

# Label dan Entry untuk Hasil
label_hasil = ttk.Label(input_frame, text="Hasil : ", foreground='red', font=4)
label_hasil.grid(column=0, row=7, padx=5, pady=5, sticky='W')

label_hasil = ttk.Label(input_frame, text="Network 1 : ", font=1)
label_hasil.grid(column=0, row=8, padx=5, pady=5, sticky='W')

# Text untuk menampilkan jaringan
network_text = tk.Text(input_frame, height=8, width=30)
network_text.grid(column=0, row=9, columnspan=2, padx=5, pady=5, sticky='W')


# Menambahkan teks ke dalam Text
network_text.insert(tk.END, "Network 1 - Pass DC Current\n")
network_text.insert(tk.END, "  o─── L ───┬────────┐\n")
network_text.insert(tk.END, "            │        │\n")
network_text.insert(tk.END, "            │        │\n")
network_text.insert(tk.END, "Z0          C       ZL\n")
network_text.insert(tk.END, "            │        │\n")
network_text.insert(tk.END, "            │        │\n")
network_text.insert(tk.END, "  o─────────┴────────┘\n")
network_text.insert(tk.END, "\n---------------------\n")

# Label dan Entry untuk nilai induktor
label_induktor = ttk.Label(input_frame, text="Nilai Induktor :", font=2)
label_induktor.grid(column=0, row=19, padx=5, pady=5, sticky='W')


label_capasitor = ttk.Label(input_frame, text="Nilai Capasitor :", font=2)
label_capasitor.grid(column=0, row=20, padx=5, pady=5, sticky='W')

label_hasil = ttk.Label(input_frame, text="Network 2 : ", font=1)
label_hasil.grid(column=0, row=21, padx=5, pady=5, sticky='W')

# Text untuk menampilkan jaringan
network_text = tk.Text(input_frame, height=8, width=30)
network_text.grid(column=0, row=22, columnspan=2, padx=5, pady=5, sticky='W')


# Menambahkan teks ke dalam Text
network_text.insert(tk.END, "Network 2 - Block DC Current\n")
network_text.insert(tk.END, "  o───────┬─── C ────┐\n")
network_text.insert(tk.END, "          │          │\n")
network_text.insert(tk.END, "          │          │\n")
network_text.insert(tk.END, "Z0        L         ZL\n")
network_text.insert(tk.END, "          │          │\n")
network_text.insert(tk.END, "          │          │\n")
network_text.insert(tk.END, "  o───────┴──────────┘\n")
network_text.insert(tk.END, "\n---------------------\n")

# Label dan Entry untuk nilai induktor
label_induktor = ttk.Label(input_frame, text="Nilai Induktor :", font=2)
label_induktor.grid(column=0, row=32, padx=5, pady=5, sticky='W')


label_capasitor = ttk.Label(input_frame, text="Nilai Capasitor :", font=2)
label_capasitor.grid(column=0, row=33, padx=5, pady=5, sticky='W')

# Initialize a blank screen
fig, ax = plt.subplots(figsize=(12, 8))  # Mengatur ukuran figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(column=2, row=0, rowspan=10, padx=15, pady=15)
canvas.draw()

root.mainloop()
