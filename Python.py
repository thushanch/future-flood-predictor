import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import sys
import threading
import pandas as pd
import numpy as np
import pymannkendall as mk
from pyextremes import EVA
import lmoments3.distr as ld
from scipy.stats import kstest
from nsevd import GEV
import matplotlib

# Use 'Agg' backend for matplotlib to prevent it from
# trying to open its own GUI window, which conflicts with Tkinter.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- A helper class to redirect stdout (print statements) to the GUI ---
class TextRedirector:
    """A helper class to redirect stdout to a Tkinter Text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, s):
        def write_to_widget():
            self.widget.configure(state="normal")
            self.widget.insert(tk.END, s, (self.tag,))
            self.widget.see(tk.END)  # Auto-scroll
            self.widget.configure(state="disabled")
        
        # Ensure GUI updates happen on the main thread
        self.widget.after(0, write_to_widget)

    def flush(self):
        pass  # Required for file-like object

# ######################################################################
# ### MODULES 1-5: ANALYSIS BACKEND (FROM TECHNICAL GUIDE)
# ######################################################################

def load_series_from_csv(filepath, file_type):
    """
    Module 1: Loads and parses streamflow data from user-specified CSV formats.
    
    :param filepath: str, path to the CSV file
    :param file_type: str, one of ['annual_ams', 'daily', 'subdaily']
    :return: (pandas.Series, str), tuple of the time series and data type
    """
    print(f"Loading data from {filepath} (format: {file_type})")
    
    if file_type == 'annual_ams':
        # Format: year|highest flood (e.g., "2001,1500")
        series = pd.read_csv(
            filepath, 
            index_col=0, 
            parse_dates=True,
            header=None,
            names=['date', 'streamflow']
        ).squeeze("columns")
        series.index = pd.to_datetime(series.index, format='%Y')
        series.name = "Annual Max Streamflow"
        print("Successfully loaded pre-extracted AMS data.")
        return series, 'AMS'

    elif file_type == 'daily':
        # Format: day|streamflow (e.g., "2001-01-01,120.5")
        series = pd.read_csv(
            filepath,
            parse_dates=[0],  # Parse the first column as dates
            index_col=0,      # Use the first column as the index
            header=None,
            names=['date', 'streamflow']
        ).squeeze("columns")
        series.name = "Daily Streamflow"
        print("Successfully loaded daily data.")
        return series, 'raw'

    elif file_type == 'subdaily':
        # Format: day|time(hh:mm)|streamflow (e.g., "2001-01-01,08:00,110.2")
        try:
            series = pd.read_csv(
                filepath,
                parse_dates={'datetime': [0, 1]}, # Combine first two columns
                index_col='datetime',
                header=None,
                names=['day', 'time', 'streamflow']
            ).squeeze("columns")
        except Exception as e:
            print(f"Could not parse 3-col subdaily, trying 2-col... Error: {e}")
            # Fallback: day|streamflow (where day is a full timestamp)
            series = pd.read_csv(
                filepath,
                parse_dates=[0],
                index_col=0,
                header=None,
                names=['datetime', 'streamflow']
            ).squeeze("columns")
            
        series.name = "Sub-daily Streamflow"
        print("Successfully loaded sub-daily data.")
        return series, 'raw'
    else:
        raise ValueError("Invalid file_type. Must be 'annual_ams', 'daily', or 'subdaily'.")

def check_stationarity(extreme_series):
    """
    Module 2: Performs the Hamed and Rao Modified Mann-Kendall test
    to check for trends in the presence of serial correlation.
    
    :param extreme_series: pandas.Series of extreme values (e.g., AMS)
    :return: bool, True if a significant trend is detected (non-stationary)
    """
    print("\n--- Checking for Non-Stationary Trends ---")
    
    if len(extreme_series) < 10:
        print("Warning: Data series is too short (<10 points) for trend test. Assuming stationary.")
        return False
        
    result = mk.hamed_rao_modification_test(extreme_series.values)
    
    print(f"Mann-Kendall Test (Hamed & Rao Modification):")
    print(f"  Trend: {result.trend}")
    print(f"  Significant (h): {result.h}")
    print(f"  P-value (p): {result.p:.4f}")
    
    if result.h:
        print("\nWARNING: A significant {result.trend} trend was detected.")
        print("Stationary FFA is not appropriate.")
        print("Proceeding with Non-Stationary analysis is required.")
    else:
        print("\nSUCCESS: Data appears stationary.")
        print("Proceeding with Stationary analysis.")
        
    return result.h  # True if trend is present

def run_stationary_analysis_pyextremes(data, data_type, method='AMS', 
                                         pot_threshold=None, pot_window='3D'):
    """
    Module 3: Runs a full stationary EVA using the pyextremes library.
    NOTE: This is an optional analysis path, not fully integrated for brevity.
    The main logic uses Module 4 (lmoments).
    """
    print(f"\n--- Running Stationary Analysis ({method}) using pyextremes ---")
    
    if data_type == 'AMS' and method == 'AMS':
        print("Initializing model from pre-extracted Annual Maxima...")
        model = EVA.from_extremes(data, extremes_method="BM")
    elif data_type == 'raw':
        print("Initializing model from raw time series...")
        model = EVA(data)
        if method == 'AMS':
            print("Extracting Annual Maxima (Block Maxima)...")
            model.get_extremes(method="BM", block_size="365.2425D")
        elif method == 'POT':
            if pot_threshold is None:
                raise ValueError("pot_threshold must be set for POT method.")
            print(f"Extracting Peaks-Over-Threshold (POT)...")
            model.get_extremes(method="POT", threshold=pot_threshold, r=pd.to_timedelta(pot_window))
    else:
        raise ValueError("Invalid data/method combination.")

    print("Fitting statistical model (GEV or GP)...")
    model.fit_model()
    
    return_periods = [2, 5, 10, 25, 50, 100, 200, 500]
    summary = model.get_summary(return_period=return_periods, alpha=0.95)
    
    print("\n--- Flood Frequency Results (pyextremes) ---")
    print(summary)
    
    # We comment out the plot_diagnostic call as it will
    # interfere with the Tkinter mainloop.
    # fig, axes = model.plot_diagnostic(alpha=0.95)
    # plt.show()
    
    return model, summary

def fit_lp3_l_moments(ams_series):
    """
    Module 4a: Fits a Log-Pearson III distribution using L-Moments.
    """
    print("\n--- Fitting Log-Pearson III (L-Moments) ---")
    
    log_data = np.log(ams_series.values)
    
    try:
        params = ld.pe3.lmom_fit(log_data)
        fitted_dist = ld.pe3(**params)
        
        print(f"Log-Pearson III Parameters (on log-data):")
        print(f"  Skew: {params['skew']:.4f}")
        print(f"  Location (mu): {params['loc']:.4f}")
        print(f"  Scale (sigma): {params['scale']:.4f}")
        
        stat, p_val = kstest(log_data, fitted_dist.cdf)
        print(f"K-S Goodness-of-Fit: p-value = {p_val:.4f}")
        if p_val < 0.05:
            print("Warning: Model may be a poor fit to the data.")
            
        return fitted_dist, True # True indicates it's a log-distribution
    except Exception as e:
        print(f"Error fitting LP3: {e}")
        return None, False


def fit_gev_l_moments(ams_series):
    """
    Module 4b: Fits a Generalized Extreme Value (GEV) distribution using L-Moments.
    """
    print("\n--- Fitting GEV (L-Moments) ---")
    
    try:
        params = ld.gev.lmom_fit(ams_series.values)
        fitted_dist = ld.gev(**params)

        print(f"GEV Parameters:")
        print(f"  Shape (c): {params['c']:.4f}")
        print(f"  Location (loc): {params['loc']:.4f}")
        print(f"  Scale (scale): {params['scale']:.4f}")
        
        stat, p_val = kstest(ams_series.values, fitted_dist.cdf)
        print(f"K-S Goodness-of-Fit: p-value = {p_val:.4f}")
        if p_val < 0.05:
            print("Warning: Model may be a poor fit to the data.")
            
        return fitted_dist, False # False, not a log-distribution
    except Exception as e:
        print(f"Error fitting GEV: {e}")
        return None, False


def get_quantiles_from_model(fitted_dist, is_log_dist, aeps=[0.5, 0.2, 0.1, 0.04, 0.02, 0.01, 0.005, 0.002]):
    """
    Module 4c: Calculates flood quantiles (magnitudes) for given AEPs.
    Returns a dictionary of results.
    """
    if fitted_dist is None:
        return {}
        
    print("\n--- Calculating Flood Quantiles ---")
    print(f"{'ARI (Year)':<12} {'AEP (%)':<8} {'Discharge (m³/s)':<15}")
    print("-" * 40)
    
    results = {}
    for aep in aeps:
        ari = 1 / aep
        prob = 1 - aep
        
        q_fitted = fitted_dist.ppf(prob)
        
        if is_log_dist:
            q_final = np.exp(q_fitted)
        else:
            q_final = q_fitted
            
        print(f"{ari:<12.0f} {aep*100:<8.1f} {q_final:<15.2f}")
        results[ari] = q_final
        
    return results

def run_non_stationary_analysis(ams_series):
    """
    Module 5a: Fits a non-stationary GEV model where flood parameters
    evolve with time.
    """
    print("\n--- Running Non-Stationary GEV Analysis ---")
    
    data = ams_series.values
    time_covariate = np.arange(len(data))
    
    covariates = {
        'location': time_covariate,
        'scale': time_covariate,
        'shape': None  # None = stationary (constant) parameter
    }

    try:
        model_ns = GEV(data, covariates=covariates)
        
        print("Fitting non-stationary model (this may take a moment)...")
        model_ns.fit(method='mle')
        
        print("\n--- Non-Stationary Model Fit Results ---")
        print(model_ns.get_parameters())
        
        return model_ns
    except Exception as e:
        print(f"Error fitting non-stationary model: {e}")
        return None

def get_future_flood_levels(model_ns, start_year, 
                            future_years=[2030, 2040, 2050, 2075, 2100], aep=0.01):
    """
    Module 5b: Calculates the 100-year (1% AEP) flood level for future years.
    Returns a dictionary of results.
    """
    if model_ns is None:
        return {}
        
    ari = 1 / aep
    print(f"\n--- Projected Future {ari:.0f}-Year Flood Levels (1% AEP) ---")
    
    results = {}
    for year in future_years:
        try:
            time_index = year - start_year
            
            q_future = model_ns.get_return_level(aep=aep, time=time_index)
            print(f"  {year}: {q_future:.2f} m³/s")
            results[year] = q_future
        except Exception as e:
            print(f"Error projecting for year {year}: {e}")
            results[year] = np.nan
            
    return results

# ######################################################################
# ### MODULE 6: MAIN CONTROLLER CLASS (ADAPTED FOR GUI)
# ######################################################################

class FloodAnalyzer:
    """Module 6: Ties all modules together."""
    
    def __init__(self, filepath, file_type):
        self.series, self.data_type = load_series_from_csv(filepath, file_type)
        self.ams_series = None
        self.is_non_stationary = None
    
    def prepare_ams_data(self):
        if self.data_type == 'AMS':
            self.ams_series = self.series
        else:
            print("Extracting Annual Maximum Series from raw data...")
            self.ams_series = self.series.resample('A').max()
            self.ams_series = self.ams_series.dropna()
        
        print(f"Extracted {len(self.ams_series)} years of AMS data.")
        
    def run_analysis(self):
        stationary_results = {'lp3': {}, 'gev': {}}
        non_stationary_results = {}

        if self.ams_series is None:
            self.prepare_ams_data()
            
        if self.ams_series.empty or len(self.ams_series) < 10:
             print("\nERROR: Not enough AMS data to perform analysis (minimum 10 years).")
             return stationary_results, non_stationary_results
            
        self.is_non_stationary = check_stationarity(self.ams_series)
        
        if self.is_non_stationary:
            print("\n*** ACTION: Trend detected. Running Non-Stationary models. ***")
            model_ns = run_non_stationary_analysis(self.ams_series)
            
            if model_ns:
                start_year = self.ams_series.index.year.min()
                non_stationary_results = get_future_flood_levels(model_ns, start_year)
            
        else:
            print("\n*** ACTION: No trend detected. Running Stationary models. ***")
            
            # Run the LP3 model
            lp3_dist, is_log_lp3 = fit_lp3_l_moments(self.ams_series)
            if lp3_dist:
                stationary_results['lp3'] = get_quantiles_from_model(lp3_dist, is_log_lp3)
            
            # Run the GEV model
            gev_dist, is_log_gev = fit_gev_l_moments(self.ams_series)
            if gev_dist:
                stationary_results['gev'] = get_quantiles_from_model(gev_dist, is_log_gev)
            
        print("\n--- Analysis Complete ---")
        return stationary_results, non_stationary_results


# ######################################################################
# ### MAIN GUI APPLICATION CLASS
# ######################################################################

class FloodPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flood Frequency Analysis Tool")
        self.root.geometry("800x700")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10, 'bold'))
        self.style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))

        # --- Variables ---
        self.filepath_var = tk.StringVar()
        self.filetype_var = tk.StringVar(value='daily')
        self.analyzer = None
        self.analysis_thread = None

        # --- Main Frame ---
        self.main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Title ---
        self.title_label = ttk.Label(self.main_frame, text="Flood Frequency Analysis Tool", style="Header.TLabel")
        self.title_label.pack(pady=(5, 20))

        # --- Input Frame ---
        self.input_frame = ttk.Frame(self.main_frame, padding=15, borderwidth=2, relief="groove")
        self.input_frame.pack(fill=tk.X, pady=10)
        self.input_frame.columnconfigure(1, weight=1)

        self.file_label = ttk.Label(self.input_frame, text="Data File:")
        self.file_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.file_entry = ttk.Entry(self.input_frame, textvariable=self.filepath_var, state="readonly", width=60)
        self.file_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        self.browse_button = ttk.Button(self.input_frame, text="Browse...", command=self.load_file)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.type_label = ttk.Label(self.input_frame, text="File Type:")
        self.type_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.type_combo = ttk.Combobox(
            self.input_frame,
            textvariable=self.filetype_var,
            values=['daily', 'subdaily', 'annual_ams'],
            state="readonly"
        )
        self.type_combo.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        
        # --- Control Button ---
        self.run_button = ttk.Button(
            self.main_frame,
            text="Run Full Analysis",
            command=self.run_analysis_thread,
            style="TButton"
        )
        self.run_button.pack(pady=15, ipadx=10, ipady=5)

        # --- Output Notebook (Tabs) ---
        self.output_notebook = ttk.Notebook(self.main_frame)
        self.output_notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # --- Tab 1: Log ---
        self.log_frame = ttk.Frame(self.output_notebook, padding=10)
        self.log_text = tk.Text(self.log_frame, height=10, width=80, state="disabled", wrap=tk.WORD, font=('Courier', 9))
        self.log_scroll = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = self.log_scroll.set
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.output_notebook.add(self.log_frame, text="Analysis Log")

        # --- Tab 2: Stationary Results ---
        self.stat_frame = ttk.Frame(self.output_notebook, padding=10)
        self.stat_tree = ttk.Treeview(
            self.stat_frame,
            columns=("Model", "ARI", "AEP", "Discharge"),
            show="headings"
        )
        self.stat_tree.heading("Model", text="Model")
        self.stat_tree.heading("ARI", text="ARI (Year)")
        self.stat_tree.heading("AEP", text="AEP (%)")
        self.stat_tree.heading("Discharge", text="Discharge (m³/s)")
        self.stat_tree.column("Model", width=100)
        self.stat_tree.column("ARI", width=100)
        self.stat_tree.column("AEP", width=100)
        self.stat_tree.column("Discharge", width=150)
        self.stat_tree.pack(fill=tk.BOTH, expand=True)
        self.output_notebook.add(self.stat_frame, text="Stationary Results")
        
        # --- Tab 3: Non-Stationary Projections ---
        self.nonstat_frame = ttk.Frame(self.output_notebook, padding=10)
        self.nonstat_tree = ttk.Treeview(
            self.nonstat_frame,
            columns=("Year", "Discharge"),
            show="headings"
        )
        self.nonstat_tree.heading("Year", text="Projection Year")
        self.nonstat_tree.heading("Discharge", text="Projected Discharge (1% AEP)")
        self.nonstat_tree.column("Year", width=150)
        self.nonstat_tree.column("Discharge", width=200)
        self.nonstat_tree.pack(fill=tk.BOTH, expand=True)
        self.output_notebook.add(self.nonstat_frame, text="Non-Stationary Projections")

        # --- Menu Bar ---
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="New Analysis", command=self.reset_analysis)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="About", command=self.show_about)

        # --- Redirect stdout ---
        self.redirector = TextRedirector(self.log_text)
        sys.stdout = self.redirector

    def load_file(self):
        """Opens a file dialog to select a CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select Streamflow Data File",
            filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filepath:
            self.filepath_var.set(filepath)
            print(f"Selected file: {filepath}\n")

    def reset_analysis(self):
        """Resets all fields for a new analysis."""
        self.filepath_var.set("")
        self.filetype_var.set("daily")
        self.analyzer = None
        if self.analysis_thread and self.analysis_thread.is_alive():
            print("Warning: Cannot reset, analysis is still running.")
            return

        # Clear log
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")
        
        # Clear trees
        for item in self.stat_tree.get_children():
            self.stat_tree.delete(item)
        for item in self.nonstat_tree.get_children():
            self.nonstat_tree.delete(item)
        
        print("--- Cleared for new analysis ---")

    def run_analysis_thread(self):
        """Runs the analysis in a separate thread to avoid freezing the GUI."""
        if not self.filepath_var.get():
            messagebox.showerror("Error", "Please select a data file first.")
            return

        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Busy", "An analysis is already in progress. Please wait.")
            return

        # Clear previous results
        self.reset_analysis()
        print("Starting new analysis...")
        
        self.run_button.config(state="disabled", text="Running...")
        
        self.analysis_thread = threading.Thread(
            target=self.run_analysis_logic,
            daemon=True
        )
        self.analysis_thread.start()

    def run_analysis_logic(self):
        """The actual analysis logic that runs in the thread."""
        try:
            filepath = self.filepath_var.get()
            filetype = self.filetype_var.get()
            
            self.analyzer = FloodAnalyzer(filepath, filetype)
            stat_results, nonstat_results = self.analyzer.run_analysis()
            
            # Schedule GUI updates back on the main thread
            self.root.after(0, self.populate_stationary_tree, stat_results)
            self.root.after(0, self.populate_nonstationary_tree, nonstat_results)
            
        except Exception as e:
            print(f"\n--- AN ERROR OCCURRED ---")
            print(f"Error details: {e}")
            messagebox.showerror("Analysis Error", f"An error occurred: {e}")
        finally:
            # Re-enable the button on the main thread
            self.root.after(0, lambda: self.run_button.config(state="normal", text="Run Full Analysis"))

    def populate_stationary_tree(self, stat_results):
        """Populates the stationary results table."""
        for item in self.stat_tree.get_children():
            self.stat_tree.delete(item)
            
        if not stat_results['lp3'] and not stat_results['gev']:
            return # No data to show
            
        self.output_notebook.select(self.stat_frame) # Switch to this tab
        
        for model_name, results in stat_results.items():
            if not results:
                continue
            
            model_label = "Log-Pearson III" if model_name == 'lp3' else "GEV"
            for ari, discharge in results.items():
                aep = (1 / ari) * 100
                self.stat_tree.insert(
                    "", tk.END, 
                    values=(model_label, f"{ari:.0f}", f"{aep:.2f}%", f"{discharge:.2f}")
                )

    def populate_nonstationary_tree(self, nonstat_results):
        """Populates the non-stationary results table."""
        for item in self.nonstat_tree.get_children():
            self.nonstat_tree.delete(item)
            
        if not nonstat_results:
            return # No data to show

        self.output_notebook.select(self.nonstat_frame) # Switch to this tab

        for year, discharge in nonstat_results.items():
            self.nonstat_tree.insert(
                "", tk.END,
                values=(year, f"{discharge:.2f}")
            )

    def show_about(self):
        """Shows an about message box."""
        messagebox.showinfo(
            "About Flood Frequency Analysis Tool",
            "FFA Tool v1.0\n\n"
            "This application combines a Tkinter GUI with a Python "
            "backend for stationary and non-stationary flood frequency analysis.\n\n"
            "Libraries Used:\n"
            "- pandas\n- pyMannKendall\n- lmoments3\n- nsevd\n- pyextremes"
        )

if __name__ == "__main__":
    main_root = tk.Tk()
    app = FloodPredictionApp(main_root)
    
    # We must restore stdout when the app closes
    def on_closing():
        sys.stdout = sys.__stdout__
        main_root.destroy()

    main_root.protocol("WM_DELETE_WINDOW", on_closing)
    main_root.mainloop()
