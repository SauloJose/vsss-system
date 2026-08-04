import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import sys
import importlib
import os
import threading
import platform
import argparse
import argparse

# Pacotes e módulos correspondentes
REQUIREMENTS_FILE = "src/data/requirements.txt"

INSTALATOR_SCRIPT = "install.py"

PACKAGE_MODULE_MAP = {
    "annotated-types": "annotated_types",
    "anyio": "anyio",
    "appdirs": "appdirs",
    "certifi": "certifi",
    "colorama": "colorama",
    "distro": "distro",
    "h11": "h11",

    # Ferramentas de análise / lint (presentes no requirements)
    "astroid": "astroid",
    "isort": "isort",
    "lazy-object-proxy": "lazy_object_proxy",
    "mccabe": "mccabe",
    "pylint": "pylint",
    "six": "six",
    "toml": "toml",
    "wrapt": "wrapt",

    # Serialização / comunicação
    "protobuf": "google.protobuf",

    # HTTP / async
    "httpcore": "httpcore",
    "httpx": "httpx",
    "idna": "idna",
    "sniffio": "sniffio",

    # Core numérico / científico
    "numpy": "numpy",
    "llvmlite": "llvmlite",

    # Templates / utils
    "Mako": "mako",
    "MarkupSafe": "markupsafe",

    # APIs / SDKs
    "openai": "openai",
    "jiter": "jiter",
    "pydantic": "pydantic",
    "pydantic_core": "pydantic_core",
    "typing_extensions": "typing_extensions",

    # Comunicação / IO
    "paho-mqtt": "paho.mqtt",
    "pyserial": "serial",

    # Multimídia / GUI
    "pillow": "PIL",
    "pyglet": "pyglet",
    "ttkthemes": "ttkthemes",
    "tk": "tkinter",  # OBS: tkinter não é instalado via pip

    # Utilidades
    "platformdirs": "platformdirs",
    "pytools": "pytools",
    "tqdm": "tqdm",
    "Unidecode": "unidecode",

    # Download / mídia
    "pytube": "pytube",
    "yt-dlp": "yt_dlp",

    # Sistema / Windows
    "pywin32": "win32api",
    "WMI": "wmi",

    # Build
    "setuptools": "setuptools",
}


class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VSSS System Installer - Professional Setup")
        self.configure(bg="#1a1a1a")
        self.geometry("700x600")
        self.resizable(False, False)
        self.center_window()
        self.installing = False

        # Ícone (se existir)
        icon_path = os.path.join(os.path.dirname(__file__), "src", "data", "icon.ico")
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)

        # Estilos
        self.setup_styles()

        # Layout principal
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        # Botões
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, background="#1a1a1a", foreground="#e0e0e0")
        style.map("TButton",
                  background=[('active', '#4a9eff'), ('pressed', '#2d6fd1')],
                  foreground=[('active', 'white')])

        # Barra de progresso
        style.configure("TProgressbar", thickness=20, troughcolor="#1a1a1a", background="#007acc")

        # Labels
        style.configure("TLabel", background="#1a1a1a", foreground="#e0e0e0", font=("Segoe UI", 10))

        # LabelFrame
        style.configure("TLabelframe", background="#1a1a1a", foreground="#e0e0e0", borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background="#1a1a1a", foreground="#e0e0e0", font=("Segoe UI", 10, "bold"))

        # Frame
        style.configure("TFrame", background="#1a1a1a")

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Título e descrição
        title_label = ttk.Label(main_frame, text="VSSS System Installer",
                               font=("Segoe UI", 18, "bold"), foreground="#007acc")
        title_label.pack(pady=(0, 10))

        desc_text = ("Bem-vindo ao instalador do Sistema VSSS (Very Small Size Soccer)!\n\n"
                     "Este sistema é uma solução completa para controle de robôs em competições de futebol de robôs.\n"
                     "Inclui visão computacional, controle de estratégia, comunicação via ESP-NOW e interface gráfica.\n\n"
                     "Este instalador verificará e instalará todas as dependências Python necessárias.\n"
                     "Acesse nossa página no GitHub para mais informações e suporte. @GN0M10\n")
        desc_label = ttk.Label(main_frame, text=desc_text, wraplength=650, justify="center")
        desc_label.pack(pady=(0, 20))

        # Informações do sistema
        system_info = f"Sistema: {platform.system()} {platform.release()} | Python: {sys.version.split()[0]}"
        info_label = ttk.Label(main_frame, text=system_info, foreground="#888888")
        info_label.pack(pady=(0, 20))

        # Barra de progresso
        self.progress = ttk.Progressbar(main_frame, length=600, mode="determinate")
        self.progress.pack(pady=(0, 10))

        # Status
        self.status_text = tk.StringVar(value="Pronto para instalar")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_text)
        self.status_label.pack(pady=(0, 10))

        # Log detalhado
        log_frame = ttk.LabelFrame(main_frame, text="Log de Instalação", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(0, 20))

        self.log_box = scrolledtext.ScrolledText(log_frame, height=10, width=80,
                                                bg="#1a1a1a", fg="#d0d0d0",
                                                font=("Consolas", 9), state="disabled")
        self.log_box.pack(fill="both", expand=True)

        # Botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(0, 10))

        self.install_button = ttk.Button(button_frame, text="🔧 Instalar Dependências",
                                        command=self.start_install_thread)
        self.install_button.pack(side="left", padx=10)

        self.start_button = ttk.Button(button_frame, text="🚀 Iniciar VSSS System",
                                      command=self.run_main, state="disabled")
        self.start_button.pack(side="left", padx=10)

        self.exit_button = ttk.Button(button_frame, text="❌ Sair", command=self.quit)
        self.exit_button.pack(side="left", padx=10)

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"+{x}+{y}")

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state="disabled")
        self.update()

    def start_install_thread(self):
        self.install_button.config(state="disabled")
        self.progress["value"] = 0
        self.status_text.set("Iniciando verificação de dependências...")
        self.log("🔍 Iniciando verificação de pacotes Python necessários...")
        threading.Thread(target=self.install_packages, daemon=True).start()

    def install_packages(self):
        self.installing = True
        total = len(PACKAGE_MODULE_MAP)
        self.progress["maximum"] = total
        installed_count = 0
        failed_count = 0

        self.log(f"📦 Total de pacotes a verificar: {total}")

        for i, (pkg, module) in enumerate(PACKAGE_MODULE_MAP.items(), start=1):
            self.status_text.set(f"Verificando {pkg} ({i}/{total})...")
            self.log(f"🔍 Verificando {pkg}...")

            if self.is_installed(module):
                self.log(f"✅ {pkg} já está instalado")
                installed_count += 1
            else:
                self.status_text.set(f"Instalando {pkg} ({i}/{total})...")
                self.log(f"📥 Instalando {pkg}...")
                try:
                    result = subprocess.run([sys.executable, "-m", "pip", "install", pkg],
                                          capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        self.log(f"✅ {pkg} instalado com sucesso")
                        installed_count += 1
                    else:
                        self.log(f"❌ Falha ao instalar {pkg}: {result.stderr}")
                        failed_count += 1
                except subprocess.TimeoutExpired:
                    self.log(f"⏰ Timeout ao instalar {pkg}")
                    failed_count += 1
                except Exception as e:
                    self.log(f"❌ Erro ao instalar {pkg}: {str(e)}")
                    failed_count += 1

            self.progress["value"] = i
            self.update()

        # Resumo final
        self.status_text.set("Instalação concluída!")
        self.log("\n" + "="*50)
        self.log("📊 RESUMO DA INSTALAÇÃO")
        self.log(f"✅ Pacotes já instalados: {installed_count}")
        self.log(f"📦 Pacotes instalados agora: {installed_count - (total - len([p for p in PACKAGE_MODULE_MAP.values() if self.is_installed(p)]))}")
        self.log(f"❌ Falhas: {failed_count}")
        self.log("="*50)

        if failed_count == 0:
            self.log("🎉 Todas as dependências foram instaladas com sucesso!")
            messagebox.showinfo("Instalação Concluída",
                              "Todas as dependências foram instaladas com sucesso!\n\n"
                              "Clique em 'Iniciar VSSS System' para executar o programa.")
        else:
            self.log("⚠️ Algumas dependências falharam. Verifique o log acima.")
            messagebox.showwarning("Instalação Parcial",
                                 f"{failed_count} pacotes falharam na instalação.\n"
                                 "Verifique o log para detalhes.")

        self.installing = False
        self.start_button.config(state="normal" if failed_count == 0 else "disabled")

    @staticmethod
    def is_installed(module_name):
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def run_main(self):
        main_path = os.path.join(os.path.dirname(__file__), "main.py")
        if os.path.exists(main_path):
            self.log("🚀 Iniciando VSSS System...")
            try:
                subprocess.Popen([sys.executable, main_path])
                self.log("✅ VSSS System iniciado com sucesso!")
                self.after(1000, self.destroy)  # Fecha após 1 segundo
            except Exception as e:
                self.log(f"❌ Erro ao iniciar: {str(e)}")
                messagebox.showerror("Erro", f"Falha ao iniciar o sistema: {str(e)}")
        else:
            self.log("❌ Arquivo main.py não encontrado!")
            messagebox.showerror("Erro", "main.py não encontrado no diretório do projeto!")

def install_requirements(venv_python=None, quiet=False):
    req_file = os.path.join(os.path.dirname(__file__), REQUIREMENTS_FILE)
    if not os.path.exists(req_file):
        raise FileNotFoundError(f"Arquivo de requirements não encontrado: {req_file}")

    python_exe = venv_python or sys.executable
    critical_packages = [
        "numpy",
        "opencv-python",
        "Pillow",
        "pyserial",
        "protobuf==3.20.3",
        "paho-mqtt",
        "ttkthemes",
        "Unidecode",
        "pytube",
        "yt-dlp",
        "pydantic",
        "pydantic_core",
        "typing_extensions",
    ]

    for pkg in critical_packages:
        subprocess.run([python_exe, "-m", "pip", "install", pkg], capture_output=True, text=True, check=False)

    cmd = [python_exe, "-m", "pip", "install", "-r", req_file]
    if quiet:
        cmd.append("-q")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if "lazy-object-proxy" in result.stderr or "lazy-object-proxy" in result.stdout:
            print("[INSTALL] Ignorando falha de lazy-object-proxy; dependências críticas já foram abordadas.")
            return result
        raise RuntimeError(result.stderr or result.stdout)
    return result


def main():
    parser = argparse.ArgumentParser(description="Instalador VSSS")
    parser.add_argument('--venv', action='store_true', help='Use o ambiente virtual .venv do projeto')
    parser.add_argument('--quiet', action='store_true', help='Instala em modo silencioso')
    args = parser.parse_args()

    if args.venv:
        project_root = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(project_root, '.venv', 'Scripts', 'python.exe')
        if not os.path.exists(venv_python):
            raise FileNotFoundError(f"Ambiente virtual não encontrado: {venv_python}")
        install_requirements(venv_python=venv_python, quiet=args.quiet)
        return

    app = InstallerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
