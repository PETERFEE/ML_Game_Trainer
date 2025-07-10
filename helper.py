# helper.py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.ion()

_figure_managers = {}
_plot_lines = {}

def plot(scores, records, mean_scores):
    fig_id = 'training_plot'

    if fig_id in _figure_managers and plt.fignum_exists(_figure_managers[fig_id].num):
        try:
            fig = _figure_managers[fig_id].canvas.figure
            plt.figure(fig.number)

            _plot_lines[fig_id]['scores'].set_xdata(range(len(scores)))
            _plot_lines[fig_id]['scores'].set_ydata(scores)
            _plot_lines[fig_id]['records'].set_xdata(range(len(records)))
            _plot_lines[fig_id]['records'].set_ydata(records)
            _plot_lines[fig_id]['mean_scores'].set_xdata(range(len(mean_scores)))
            _plot_lines[fig_id]['mean_scores'].set_ydata(mean_scores)

            plt.xlim(0, len(scores))
            current_max_y = 10
            if scores:
                current_max_y = max(current_max_y, max(scores))
            if records:
                current_max_y = max(current_max_y, max(records))
            if mean_scores:
                current_max_y = max(current_max_y, max(mean_scores))
            plt.ylim(ymin=0, ymax=current_max_y * 1.1)
        except Exception as e:
            print(f"Error updating existing plot: {e}")
            if fig_id in _figure_managers:
                del _figure_managers[fig_id]
            if fig_id in _plot_lines:
                del _plot_lines[fig_id]

    else:
        fig = plt.figure(figsize=(8, 6))
        _figure_managers[fig_id] = fig.canvas.manager
        
        line1, = plt.plot(scores, label='Scores (Current Game)')
        line2, = plt.plot(records, label='Highest Score (Record)')
        line3, = plt.plot(mean_scores, label='Average Score')
        _plot_lines[fig_id] = {'scores': line1, 'records': line2, 'mean_scores': line3}
        
        plt.title('Training Progress')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.ylim(ymin=0)

        print("Plot window created.")

    plt.show(block=False)
    
    try:
        _figure_managers[fig_id].canvas.draw_idle()
        _figure_managers[fig_id].canvas.flush_events() 
    except Exception as e:
        print(f"Error updating plot display: {e}")

    return _figure_managers[fig_id]

def toggle_plot_visibility(fig_id='training_plot'):
    if fig_id not in _figure_managers:
        print(f"Plot '{fig_id}' not found. Cannot toggle visibility.")
        return

    try:
        manager = _figure_managers[fig_id]
        if manager.window.wm_state() == 'normal' or manager.window.wm_state() == 'iconic':
            manager.window.withdraw()
            print(f"Plot '{fig_id}' hidden.")
        else:
            manager.window.deiconify()
            manager.window.lift()
            print(f"Plot '{fig_id}' shown.")
    except Exception as e:
        print(f"Error toggling plot visibility: {e}")
        if fig_id in _figure_managers:
            del _figure_managers[fig_id]
        if fig_id in _plot_lines:
            del _plot_lines[fig_id]

