# helper.py
import matplotlib
matplotlib.use('TkAgg') # Crucial: Ensures Matplotlib uses the Tkinter backend
import matplotlib.pyplot as plt

plt.ion() # Turn on interactive mode for non-blocking plots

_figure_managers = {} # Stores references to matplotlib figure managers
_plot_lines = {}      # Stores references to the actual plot lines for updating

# Changed function signature to accept scores, records, and mean_scores
def plot(scores, records, mean_scores): # <--- UPDATED
    fig_id = 'training_plot' # Unique ID for your plot window

    if fig_id in _figure_managers and plt.fignum_exists(_figure_managers[fig_id].num):
        fig = _figure_managers[fig_id].canvas.figure
        plt.figure(fig.number) # Make this the current figure for matplotlib operations

        # Update existing plot data
        _plot_lines[fig_id]['scores'].set_xdata(range(len(scores)))
        _plot_lines[fig_id]['scores'].set_ydata(scores)
        _plot_lines[fig_id]['records'].set_xdata(range(len(records)))
        _plot_lines[fig_id]['records'].set_ydata(records)
        _plot_lines[fig_id]['mean_scores'].set_xdata(range(len(mean_scores))) # <--- NEW
        _plot_lines[fig_id]['mean_scores'].set_ydata(mean_scores) # <--- NEW

        # Autoscale axes as data changes, but keep ymin at 0
        plt.xlim(0, len(scores))
        current_max_y = 10
        if scores:
            current_max_y = max(current_max_y, max(scores))
        if records:
            current_max_y = max(current_max_y, max(records))
        if mean_scores: # <--- NEW: Consider mean_scores for ymax
            current_max_y = max(current_max_y, max(mean_scores))
        plt.ylim(ymin=0, ymax=current_max_y * 1.1)

    else:
        # Create a new figure if it doesn't exist or was closed
        fig = plt.figure(figsize=(8, 6))
        _figure_managers[fig_id] = fig.canvas.manager
        
        # Plot for the first time and store line references
        # Matplotlib's default color cycle: blue (first), orange (second), green (third)
        line1, = plt.plot(scores, label='Scores (Current Game)') # Blue
        line2, = plt.plot(records, label='Highest Score (Record)') # Orange
        line3, = plt.plot(mean_scores, label='Average Score') # <--- NEW: Green
        _plot_lines[fig_id] = {'scores': line1, 'records': line2, 'mean_scores': line3} # <--- UPDATED: Added mean_scores
        
        plt.title('Training Progress')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.legend() # Keep legend to identify which line is which
        plt.grid(True) # Keep grid for readability
        plt.ylim(ymin=0) # Ensure y-axis starts at 0

        print("Plot window created.")

    plt.show(block=False)
    
    _figure_managers[fig_id].canvas.draw_idle()
    _figure_managers[fig_id].canvas.flush_events() 

    return _figure_managers[fig_id]

def toggle_plot_visibility(fig_id='training_plot'):
    if fig_id not in _figure_managers:
        print(f"Plot '{fig_id}' not found. Cannot toggle visibility.")
        return

    manager = _figure_managers[fig_id]
    if manager.window.wm_state() == 'normal' or manager.window.wm_state() == 'iconic':
        manager.window.withdraw()
        print(f"Plot '{fig_id}' hidden.")
    else:
        manager.window.deiconify()
        manager.window.lift()
        print(f"Plot '{fig_id}' shown.")

