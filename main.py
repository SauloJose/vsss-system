import sys
import os
import importlib.util
import subprocess
import tkinter as tk
from tkinter import messagebox
from install import *

def parse_requirements(file_path):
    if not os.path.isfile(file_path):
        messagebox.showerror("Erro", f"Arquivo de requirements não encontrado:\n{file_path}")
        sys.exit(1)

    packages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pkg = line.split("==")[0]
                packages.append(pkg)
    return packages

def is_installed(package_name):
    module_name = PACKAGE_MODULE_MAP.get(package_name, package_name.replace("-", "_").lower())
    return importlib.util.find_spec(module_name) is not None

def check_missing_packages():
    packages = parse_requirements(REQUIREMENTS_FILE)
    missing = [pkg for pkg in packages if not is_installed(pkg)]
    return missing

def prompt_install():
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal
    missing = check_missing_packages()
    if missing:
        msg = "Alguns pacotes necessários não estão instalados:\n\n"
        msg += ", ".join(missing[:10])
        if len(missing) > 10:
            msg += f", ... (+{len(missing)-10} outros)"
        msg += "\n\nDeseja instalar agora?"
        if messagebox.askyesno("Dependências faltando", msg):
            root.destroy()
            # Chama o instalador
            subprocess.call([sys.executable, INSTALATOR_SCRIPT])
        else:
            messagebox.showwarning("Atenção", "O programa pode não funcionar corretamente sem os pacotes necessários.")
            root.destroy()
    else:
        root.destroy()

# Primeiro verifica e instala se necessário
prompt_install()

# Agora adiciona src/ ao path e inicia a aplicação
sys.path.append("./src/")
from app import App

app = App()
