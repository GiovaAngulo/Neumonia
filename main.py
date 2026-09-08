import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys

class PneumoniaDetector:
    def __init__(self, model_path):
        try:
            # Verificar modelo
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en:\n{model_path}")
            
            self.model = load_model(model_path)
            print("Modelo cargado correctamente!")
            
            # Configurar ventana principal con scrollbar
            self.root = tk.Tk()
            self.root.title("Detector de Neumonía - Con Scroll")
            self.root.geometry("900x750")
            
            # Crear canvas y scrollbar
            self.canvas = tk.Canvas(self.root, borderwidth=0)
            self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
            self.scrollable_frame = tk.Frame(self.canvas)
            
            # Configurar scroll
            self.scrollable_frame.bind(
                "<Configure>",
                lambda e: self.canvas.configure(
                    scrollregion=self.canvas.bbox("all")
                )
            )
            
            self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            
            self.canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")
            
            # Variables para resultados
            self.result_var = tk.StringVar(value="Seleccione una imagen")
            self.prob_var = tk.StringVar()
            
            self.create_widgets()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar:\n{str(e)}")
            sys.exit(1)
    
    def create_widgets(self):
        # Configuración de colores
        bg_color = "#f0f0f0"
        self.scrollable_frame.config(bg=bg_color)
        
        # Marco principal dentro del frame desplazable
        main_frame = tk.Frame(self.scrollable_frame, bg=bg_color)
        main_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
        
        # Botón para cargar imagen (centrado)
        btn_frame = tk.Frame(main_frame, bg=bg_color)
        btn_frame.pack(pady=20)
        
        self.load_btn = tk.Button(btn_frame, 
                                text="SELECCIONAR RADIOGRAFÍA", 
                                command=self.load_image,
                                font=("Arial", 14, "bold"),
                                bg="#4CAF50", fg="white",
                                width=30, height=2)
        self.load_btn.pack()
        
        # Área de imagen con tamaño fijo
        img_container = tk.Frame(main_frame, bg="white", bd=2, relief=tk.SUNKEN)
        img_container.pack(pady=10)
        
        self.img_label = tk.Label(img_container)
        self.img_label.pack(padx=10, pady=10)
        
        # Área de resultados destacada
        result_frame = tk.Frame(main_frame, bg=bg_color, bd=2, relief=tk.GROOVE)
        result_frame.pack(pady=20, fill=tk.X)
        
        tk.Label(result_frame, 
               text="RESULTADO DEL ANÁLISIS:", 
               font=("Arial", 16, "bold"), 
               bg=bg_color).pack(pady=(10,5))
        
        self.result_display = tk.Label(result_frame, 
                                     textvariable=self.result_var,
                                     font=("Arial", 18, "bold"),
                                     bg=bg_color)
        self.result_display.pack(pady=5)
        
        self.prob_display = tk.Label(result_frame, 
                                   textvariable=self.prob_var,
                                   font=("Arial", 14),
                                   bg=bg_color)
        self.prob_display.pack(pady=(5,10))
        
        # Botón de salida (centrado)
        exit_frame = tk.Frame(main_frame, bg=bg_color)
        exit_frame.pack(pady=20)
        
        tk.Button(exit_frame, 
                text="SALIR", 
                command=self.root.quit,
                font=("Arial", 12),
                bg="#f44336", fg="white",
                width=15).pack()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar radiografía",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Mostrar estado de carga
            self.result_var.set("Analizando...")
            self.prob_var.set("")
            self.root.update()
            
            # Verificar y cargar imagen
            img = Image.open(file_path)
            img.verify()  # Verificar integridad
            img = Image.open(file_path)  # Reabrir para uso
            
            # Procesamiento
            img_array = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img_array)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Predicción
            prediction = self.model.predict(img_array)
            prob = float(prediction[0][0]) * 100
            
            # Mostrar imagen (con tamaño máximo controlado)
            display_img = img.copy()
            display_img.thumbnail((600, 600))  # Tamaño máximo para visualización
            img_tk = ImageTk.PhotoImage(display_img)
            
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
            
            # Mostrar resultados
            if prob > 50:
                result_text = "⚠ POSIBLE NEUMONÍA DETECTADA ⚠"
                color = "red"
            else:
                result_text = "✅ RESULTADO NORMAL ✅"
                color = "green"
            
            self.result_var.set(result_text)
            self.result_display.config(fg=color)
            self.prob_var.set(f"Probabilidad: {prob:.2f}%")
            
            # Auto-ajustar scroll al inicio
            self.canvas.yview_moveto(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar:\n{str(e)}")
            self.result_var.set("Error en el análisis")
            self.prob_var.set("")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    model_path = r"C:\Users\GIOVANNY\Desktop\Neumonia\final_pneumonia_model.h5"
    
    if not os.path.exists(model_path):
        messagebox.showerror("Error", f"Archivo de modelo no encontrado:\n{model_path}")
        sys.exit(1)
    
    app = PneumoniaDetector(model_path)
    app.run()